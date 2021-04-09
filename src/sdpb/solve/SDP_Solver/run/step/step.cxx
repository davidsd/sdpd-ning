#include "../../../SDP_Solver.hxx"
#include "../../../../../Timers.hxx"
#include "../../../../../set_stream_precision.hxx"

// Tr(A B), where A and B are symmetric
El::BigFloat frobenius_product_symmetric(const Block_Diagonal_Matrix &A,
                                         const Block_Diagonal_Matrix &B);

El::BigFloat dot(const Block_Vector &a, const Block_Vector &b);

void initialize_schur_complement_solver(
  const Block_Info &block_info, const SDP &sdp,
  const Block_Diagonal_Matrix &bilinear_pairings_X_inv,
  const Block_Diagonal_Matrix &bilinear_pairings_Y, const El::Grid &block_grid,
  Block_Diagonal_Matrix &schur_complement_cholesky,
  Block_Matrix &schur_off_diagonal, El::DistMatrix<El::BigFloat> &Q,
  Timers &timers);

void compute_search_direction(
	const Block_Info &block_info, const SDP &sdp, const SDP &dsdp, const SDP_Solver &solver,
	const Block_Diagonal_Matrix &schur_complement_cholesky,
	const Block_Matrix &schur_off_diagonal,
	const Block_Diagonal_Matrix &X_cholesky,
	const Block_Diagonal_Matrix Z,
	const bool &is_compute_derivative_dBdbdc,
	const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx,
	Block_Diagonal_Matrix &dX, Block_Vector &dy, Block_Diagonal_Matrix &dY);


El::BigFloat
predictor_centering_parameter(const SDP_Solver_Parameters &parameters,
                              const bool is_primal_dual_feasible);

El::BigFloat corrector_centering_parameter(
  const SDP_Solver_Parameters &parameters, const Block_Diagonal_Matrix &X,
  const Block_Diagonal_Matrix &dX, const Block_Diagonal_Matrix &Y,
  const Block_Diagonal_Matrix &dY, const El::BigFloat &mu,
  const bool is_primal_dual_feasible, const size_t &total_num_rows);

El::BigFloat
step_length(const Block_Diagonal_Matrix &MCholesky,
            const Block_Diagonal_Matrix &dM, const El::BigFloat &gamma,
            const std::string &timer_name, Timers &timers);


void scale_multiply_add(const El::BigFloat &alpha,
	const Block_Diagonal_Matrix &A,
	const Block_Diagonal_Matrix &B,
	const El::BigFloat &beta, Block_Diagonal_Matrix &C);

// C := A B
inline void multiply(const Block_Diagonal_Matrix &A,
	const Block_Diagonal_Matrix &B, Block_Diagonal_Matrix &C)
{
	scale_multiply_add(El::BigFloat(1), A, B, El::BigFloat(0), C);
}

// X := ACholesky^{-T} ACholesky^{-1} X = A^{-1} X
void cholesky_solve(const Block_Diagonal_Matrix &ACholesky,
	Block_Diagonal_Matrix &X);

void SDP_Solver::step(const SDP_Solver_Parameters &parameters,
                      const std::size_t &total_psd_rows,
                      const bool &is_primal_and_dual_feasible,
                      const Block_Info &block_info, const SDP &sdp, const std::vector<SDP*> dsdp_list,
                      const El::Grid &grid,
                      const Block_Diagonal_Matrix &X_cholesky,
                      const Block_Diagonal_Matrix &Y_cholesky,
                      const Block_Diagonal_Matrix &bilinear_pairings_X_inv,
                      const Block_Diagonal_Matrix &bilinear_pairings_Y,
                      const Block_Vector &primal_residue_p, El::BigFloat &mu,
                      El::BigFloat &beta_corrector,
                      El::BigFloat &primal_step_length,
                      El::BigFloat &dual_step_length,
                      bool &terminate_now, Timers &timers)
{
  auto &step_timer(timers.add_and_start("run.step"));
  El::BigFloat beta_predictor;

  // Search direction: These quantities have the same structure
  // as (x, X, y, Y). They are computed twice each iteration:
  // once in the predictor step, and once in the corrector step.
  {
    // SchurComplementCholesky = L', the Cholesky decomposition of the
    // Schur complement matrix S.
    Block_Diagonal_Matrix schur_complement_cholesky(
      block_info.schur_block_sizes, block_info.block_indices,
      block_info.schur_block_sizes.size(), grid);

    // SchurOffDiagonal = L'^{-1} FreeVarMatrix, needed in solving the
    // Schur complement equation.
    Block_Matrix schur_off_diagonal;

    // Q = B' L'^{-T} L'^{-1} B' - {{0, 0}, {0, 1}}, where B' =
    // (FreeVarMatrix U).  Q is needed in the factorization of the Schur
    // complement equation.  Q has dimension N'xN', where
    //
    //   N' = cols(B) + cols(U) = N + cols(U)
    //
    // where N is the dimension of the dual objective function.  Note
    // that N' could change with each iteration.
    El::DistMatrix<El::BigFloat> Q(sdp.dual_objective_b.Height(),
                                   sdp.dual_objective_b.Height());

    // Compute SchurComplement and prepare to solve the Schur
    // complement equation for dx, dy
    initialize_schur_complement_solver(
      block_info, sdp, bilinear_pairings_X_inv, bilinear_pairings_Y, grid,
      schur_complement_cholesky, schur_off_diagonal, Q, timers);

    // Compute the complementarity mu = Tr(X Y)/X.dim
    auto &frobenius_timer(
      timers.add_and_start("run.step.frobenius_product_symmetric"));
    mu = frobenius_product_symmetric(X, Y) / total_psd_rows;
    frobenius_timer.stop();
    if(mu > parameters.max_complementarity)
      {
        terminate_now = true;
        return;
      }

    auto &predictor_timer(
      timers.add_and_start("run.step.computeSearchDirection(betaPredictor)"));


	Block_Diagonal_Matrix Z(X);
	multiply(primal_residues, Y, Z);
	cholesky_solve(X_cholesky, Z);
	Z.symmetrize();



	for (int i = 0; i < dsdp_list.size(); i++)
	{
		DSDPSOLUTION*psol = new DSDPSOLUTION(x,X,y,Y);
		compute_search_direction(block_info, sdp, *dsdp_list[i], *this, schur_complement_cholesky,
			schur_off_diagonal, X_cholesky,
			Z, parameters.compute_derivative_dBdbdc, Q, psol->dx, psol->dX, psol->dy, psol->dY);

		dsdp_sol_list.push_back(psol);
	}

    predictor_timer.stop();

	/*
    // Compute the corrector solution for (dx, dX, dy, dY)
    auto &corrector_timer(
      timers.add_and_start("run.step.computeSearchDirection(betaCorrector)"));
    beta_corrector = corrector_centering_parameter(
      parameters, X, dX, Y, dY, mu, is_primal_and_dual_feasible,
      total_psd_rows);

    compute_search_direction(block_info, sdp, *this, schur_complement_cholesky,
                             schur_off_diagonal, X_cholesky, beta_corrector,
                             mu, primal_residue_p, true, Q, dx, dX, dy, dY);
    corrector_timer.stop();
	*/
  }
  // Compute step-lengths that preserve positive definiteness of X, Y

  /*
  primal_step_length
    = step_length(X_cholesky, dX, parameters.step_length_reduction,
                  "run.step.stepLength(XCholesky)", timers);

  dual_step_length
    = step_length(Y_cholesky, dY, parameters.step_length_reduction,
                  "run.step.stepLength(YCholesky)", timers);
				  */

  /*
  // If our problem is both dual-feasible and primal-feasible,
  // ensure we're following the true Newton direction.
  if(is_primal_and_dual_feasible)
    {
      primal_step_length = El::Min(primal_step_length, dual_step_length);
      dual_step_length = primal_step_length;
    }
	*/

  /*
  // Update the primal point (x, X) += primalStepLength*(dx, dX)
  for(size_t block = 0; block < x.blocks.size(); ++block)
    {
      El::Axpy(primal_step_length, dx.blocks[block], x.blocks[block]);
    }
  dX *= primal_step_length;

  X += dX;

  // Update the dual point (y, Y) += dualStepLength*(dy, dY)
  for(size_t block = 0; block < dy.blocks.size(); ++block)
    {
      El::Axpy(dual_step_length, dy.blocks[block], y.blocks[block]);
    }
  dY *= dual_step_length;

  Y += dY;
  */

  step_timer.stop();


}


/*

auto &step_timer(timers.add_and_start("run.step"));
El::BigFloat beta_predictor;

// Search direction: These quantities have the same structure
// as (x, X, y, Y). They are computed twice each iteration:
// once in the predictor step, and once in the corrector step.
Block_Vector dx(x), dy(y);
Block_Diagonal_Matrix dX(X), dY(Y);
{
// SchurComplementCholesky = L', the Cholesky decomposition of the
// Schur complement matrix S.
Block_Diagonal_Matrix schur_complement_cholesky(
block_info.schur_block_sizes, block_info.block_indices,
block_info.schur_block_sizes.size(), grid);

// SchurOffDiagonal = L'^{-1} FreeVarMatrix, needed in solving the
// Schur complement equation.
Block_Matrix schur_off_diagonal;

// Q = B' L'^{-T} L'^{-1} B' - {{0, 0}, {0, 1}}, where B' =
// (FreeVarMatrix U).  Q is needed in the factorization of the Schur
// complement equation.  Q has dimension N'xN', where
//
//   N' = cols(B) + cols(U) = N + cols(U)
//
// where N is the dimension of the dual objective function.  Note
// that N' could change with each iteration.
El::DistMatrix<El::BigFloat> Q(sdp.dual_objective_b.Height(),
sdp.dual_objective_b.Height());

// Compute SchurComplement and prepare to solve the Schur
// complement equation for dx, dy
initialize_schur_complement_solver(
block_info, sdp, bilinear_pairings_X_inv, bilinear_pairings_Y, grid,
schur_complement_cholesky, schur_off_diagonal, Q, timers);

// Compute the complementarity mu = Tr(X Y)/X.dim
auto &frobenius_timer(
timers.add_and_start("run.step.frobenius_product_symmetric"));
mu = frobenius_product_symmetric(X, Y) / total_psd_rows;
frobenius_timer.stop();
if(mu > parameters.max_complementarity)
{
terminate_now = true;
return;
}

auto &predictor_timer(
timers.add_and_start("run.step.computeSearchDirection(betaPredictor)"));

// Compute the predictor solution for (dx, dX, dy, dY)
beta_predictor
= predictor_centering_parameter(parameters, is_primal_and_dual_feasible);
compute_search_direction(block_info, sdp, *this, schur_complement_cholesky,
schur_off_diagonal, X_cholesky, beta_predictor,
mu, primal_residue_p, false, Q, dx, dX, dy, dY);
predictor_timer.stop();

// Compute the corrector solution for (dx, dX, dy, dY)
auto &corrector_timer(
timers.add_and_start("run.step.computeSearchDirection(betaCorrector)"));
beta_corrector = corrector_centering_parameter(
parameters, X, dX, Y, dY, mu, is_primal_and_dual_feasible,
total_psd_rows);

compute_search_direction(block_info, sdp, *this, schur_complement_cholesky,
schur_off_diagonal, X_cholesky, beta_corrector,
mu, primal_residue_p, true, Q, dx, dX, dy, dY);
corrector_timer.stop();
}
// Compute step-lengths that preserve positive definiteness of X, Y
primal_step_length
= step_length(X_cholesky, dX, parameters.step_length_reduction,
"run.step.stepLength(XCholesky)", timers);

dual_step_length
= step_length(Y_cholesky, dY, parameters.step_length_reduction,
"run.step.stepLength(YCholesky)", timers);

// If our problem is both dual-feasible and primal-feasible,
// ensure we're following the true Newton direction.
if(is_primal_and_dual_feasible)
{
primal_step_length = El::Min(primal_step_length, dual_step_length);
dual_step_length = primal_step_length;
}

// Update the primal point (x, X) += primalStepLength*(dx, dX)
for(size_t block = 0; block < x.blocks.size(); ++block)
{
El::Axpy(primal_step_length, dx.blocks[block], x.blocks[block]);
}
dX *= primal_step_length;

X += dX;

// Update the dual point (y, Y) += dualStepLength*(dy, dY)
for(size_t block = 0; block < dy.blocks.size(); ++block)
{
El::Axpy(dual_step_length, dy.blocks[block], y.blocks[block]);
}
dY *= dual_step_length;

Y += dY;
step_timer.stop();


*/