//=======================================================================
// Copyright 2014-2015 David Simmons-Duffin.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "SDP_Solver.hxx"
#include "../../Timers.hxx"
#include "../../set_stream_precision.hxx"

#include <El.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>


El::BigFloat dot(const Block_Vector &a, const Block_Vector &b);

void compute_dsdp(SDP & dsdp, const Block_Info &block_info, const El::Grid &grid, const boost::filesystem::path & dsdp_filepath)
{
	SDP sdp2(dsdp_filepath, block_info, grid);

	dsdp.objective_const -= sdp2.objective_const;

	auto primal_objective_c_block(dsdp.primal_objective_c.blocks.begin());
	auto primal_objective_c2_block(sdp2.primal_objective_c.blocks.begin());
	for (auto &block_index : block_info.block_indices)
	{
		*primal_objective_c_block -= *primal_objective_c2_block;
		++primal_objective_c_block;
		++primal_objective_c2_block;
	}

	dsdp.dual_objective_b -= sdp2.dual_objective_b;

	auto free_var_matrix_block(dsdp.free_var_matrix.blocks.begin());
	auto free_var_matrix2_block(sdp2.free_var_matrix.blocks.begin());
	for (auto &block_index : block_info.block_indices)
	{
		*free_var_matrix_block -= *free_var_matrix2_block;
		++free_var_matrix_block;
		++free_var_matrix2_block;
	}

}


void compute_dsdp_mode3(SDP & dsdp, const Block_Info &block_info, const El::Grid &grid, const boost::filesystem::path & dsdp_filepath)
{
	SDP sdp2(dsdp_filepath, block_info, grid);

	dsdp.objective_const = sdp2.objective_const;

	auto primal_objective_c_block(dsdp.primal_objective_c.blocks.begin());
	auto primal_objective_c2_block(sdp2.primal_objective_c.blocks.begin());
	for (auto &block_index : block_info.block_indices)
	{
		*primal_objective_c_block = *primal_objective_c2_block;
		++primal_objective_c_block;
		++primal_objective_c2_block;
	}

	dsdp.dual_objective_b = sdp2.dual_objective_b;

	auto free_var_matrix_block(dsdp.free_var_matrix.blocks.begin());
	auto free_var_matrix2_block(sdp2.free_var_matrix.blocks.begin());
	for (auto &block_index : block_info.block_indices)
	{
		*free_var_matrix_block = *free_var_matrix2_block;
		++free_var_matrix_block;
		++free_var_matrix2_block;
	}

}



El::BigFloat compute_xBy(const Block_Info &block_info, const SDP &sdp,
	const Block_Vector &x, const Block_Vector &y)
{
	Block_Vector By(x);

	auto By_block(By.blocks.begin());
	auto primal_objective_c_block(sdp.primal_objective_c.blocks.begin());
	auto y_block(y.blocks.begin());
	auto free_var_matrix_block(sdp.free_var_matrix.blocks.begin());

	for (auto &block_index : block_info.block_indices)
	{
		// By = 0
		Zero(*By_block);
		const size_t block_size(block_info.degrees[block_index] + 1);

		// By -= FreeVarMatrix * y
		Gemm(El::Orientation::NORMAL, El::Orientation::NORMAL, El::BigFloat(1),
			*free_var_matrix_block, *y_block, El::BigFloat(1),
			*By_block);

		++y_block;
		++free_var_matrix_block;
		++By_block;
	}

	return dot(x, By);
}



El::BigFloat compute_quadratic_component(const SDP &sdp, const SDP &dsdp, const DSDPSOLUTION & sol, const SDP_Solver & solver, const Block_Info &block_info)
{

	El::BigFloat db_dy;
	if (!sol.dy.blocks.empty())
	{
		db_dy = El::Dotu(dsdp.dual_objective_b, sol.dy.blocks.front());
	}

	El::BigFloat dc_dx = dot(dsdp.primal_objective_c, sol.dx);

	El::BigFloat x_dB_dy = compute_xBy(block_info, dsdp, solver.x, sol.dy);

	El::BigFloat dx_dB_y = compute_xBy(block_info, dsdp, sol.dx, solver.y);
        
	// if (El::mpi::Rank() == 0)
	// {
        //   set_stream_precision(std::cout);
        //   std::cout << El::mpi::Rank() << " db_dy  = " << db_dy << "\n";
        //   std::cout << El::mpi::Rank() << " dc_dx  = " << dc_dx << "\n";
        //   std::cout << El::mpi::Rank() << " dx_dB_y = " << dx_dB_y << "\n";
        //   std::cout << El::mpi::Rank() << " x_dB_dy = " << x_dB_dy << "\n";
	// }

	El::BigFloat rslt = (db_dy + dc_dx - dx_dB_y - x_dB_dy)/2;

	return rslt;
}


int initialize_dsdp(SDP & sdp0, std::vector<SDP*> & dsdp_list, const Block_Info &block_info, const El::Grid &grid, const SDP_Solver_Parameters &parameters)
{
	if (El::mpi::Rank() == 0)
	{
		std::cout << " Total # of dsdp : " << parameters.list_sdp2_path.size() << "\n";
	}

	for(int i=0;i<parameters.list_sdp2_path.size();i++)
	{
		if (El::mpi::Rank() == 0)
		{
			std::cout << " Reading #" << i << " dsdp file from " << parameters.list_sdp2_path[i] << "\n";
		}

		SDP*psdp = new SDP(sdp0);

		if (parameters.compute_derivative_dBdbdc)
		{
			if (parameters.sdpd_mode_dBdbdc)
			{
				compute_dsdp_mode3(*psdp, block_info, grid, parameters.list_sdp2_path[i]);
			}
			else
				compute_dsdp(*psdp, block_info, grid, parameters.list_sdp2_path[i]);
		}

		dsdp_list.push_back(psdp);
	}
	return 1;
}

El::BigFloat compute_derivative_Balt_formula(SDP & dsdp, SDP_Solver & solver, const Block_Info &block_info)
{
	El::BigFloat dby;
	if (!solver.y.blocks.empty())
	{
		dby = dsdp.objective_const + El::Dotu(dsdp.dual_objective_b, solver.y.blocks.front());
	}

	El::BigFloat xdBy = compute_xBy(block_info, dsdp, solver.x, solver.y);

	El::BigFloat dprimalobj_Balt = dot(dsdp.primal_objective_c, solver.x) + dby - xdBy;

        // The minus sign is because (db,dc,dB) in the dsdp are minus
        // what they are in the notes. This should really be changed
        // so that no minus sign here is necessary.
	return -dprimalobj_Balt;
}

int compute_linear(std::vector<El::BigFloat> & dobj_list, std::vector<SDP*> & dsdp_list, SDP_Solver & solver, const Block_Info &block_info)
{
	for (int i = 0; i < dsdp_list.size(); i++)
	{
		dobj_list.push_back(compute_derivative_Balt_formula(*dsdp_list[i], solver, block_info));
	}
	return 1;
}

int compute_quadratic(std::vector<std::vector<El::BigFloat>> & quadratic, const SDP &sdp, const std::vector<SDP*> & dsdp_list, const SDP_Solver & solver, const Block_Info &block_info)
{
	for (int i = 0; i < dsdp_list.size(); i++)
	{
		std::vector<El::BigFloat> vec;
		for (int j = 0; j < dsdp_list.size(); j++)
		{
                  vec.push_back(compute_quadratic_component(sdp, *dsdp_list[i], *solver.dsdp_sol_list[j], solver, block_info));
		}
		quadratic.push_back(vec);
	}
	return 1;
}

Timers solve(const Block_Info &block_info, const SDP_Solver_Parameters &parameters)
{
  // Read an SDP from sdpFile and create a solver for it
  El::Grid grid(block_info.mpi_comm.value);

  Timers timers(parameters.verbosity >= Verbosity::debug);

  SDP sdp(parameters.sdp_directory, block_info, grid);

  std::vector<SDP*> dsdp_list;

  initialize_dsdp(sdp, dsdp_list, block_info, grid, parameters);

  SDP_Solver solver(parameters, block_info, grid,
                    sdp.dual_objective_b.Height());

  std::vector<El::BigFloat> dobj_list;
  compute_linear(dobj_list, dsdp_list, solver, block_info);

  SDP_Solver_Terminate_Reason reason
    = solver.run(parameters, block_info, sdp, dsdp_list, grid, timers);

  std::vector<std::vector<El::BigFloat>> quadratic;

  compute_quadratic(quadratic, sdp, dsdp_list, solver, block_info);

  if(parameters.verbosity >= Verbosity::regular && El::mpi::Rank() == 0)
    {
      set_stream_precision(std::cout);

	  if (parameters.compute_derivative_dBdbdc)
	  {
		  std::cout << "[SDPDReturnBegin.Linear]\n";
		  std::cout << "{\n";
		  for (int i = 0; i < dobj_list.size(); i++)
		  {
			  std::cout << dobj_list[i] << "\n";
			  if (i != dobj_list.size() - 1) std::cout << ",";
		  }
		  std::cout << "}\n";
		  std::cout << "[SDPDReturnEnd.Linear]\n";

		  std::cout << "[SDPDReturnBegin.Quadratic]\n";
		  std::cout << "{\n";
		  for (int i = 0; i < dsdp_list.size(); i++)
		  {
			  std::cout << "{\n";
			  for (int j = 0; j < dsdp_list.size(); j++)
			  {
				  std::cout << quadratic[i][j];
				  if (j != dsdp_list.size() - 1) std::cout << ",";
			  }
			  std::cout << "}\n";
			  if (i != dsdp_list.size() - 1) std::cout << ",";
		  }
		  std::cout << "}\n";
		  std::cout << "[SDPDReturnEnd.Quadratic]\n";

	  }
    }

  if (!parameters.no_final_checkpoint)
  {
	  solver.save_solution(reason, timers.front(), parameters.out_directory,
		  parameters.write_solution,
		  block_info.block_indices,
		  parameters.verbosity);
  }

  return timers;
}
