#include "../SDP_Solver.hxx"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/json_parser.hpp>

// We use binary checkpointing because writing text does not write all
// of the necessary digits.  The GMP library sets it to one less than
// required for round-tripping.
template <typename T>
void write_local_blocks(const T &t,
                        boost::filesystem::ofstream &checkpoint_stream)
{
  El::BigFloat zero(0);
  const size_t serialized_size(zero.SerializedSize());
  std::vector<uint8_t> local_array(serialized_size);

  for(auto &block : t.blocks)
    {
      int64_t local_height(block.LocalHeight()),
        local_width(block.LocalWidth());
      checkpoint_stream.write(reinterpret_cast<char *>(&local_height),
                              sizeof(int64_t));
      checkpoint_stream.write(reinterpret_cast<char *>(&local_width),
                              sizeof(int64_t));
      for(int64_t row = 0; row < local_height; ++row)
        for(int64_t column = 0; column < local_width; ++column)
          {
            block.GetLocal(row, column).Serialize(local_array.data());
            checkpoint_stream.write(
              reinterpret_cast<char *>(local_array.data()),
              std::streamsize(local_array.size()));
          }
    }
}

void SDP_Solver::save_checkpoint(const SDP_Solver_Parameters &parameters)
{
  const boost::filesystem::path &checkpoint_directory(
    parameters.checkpoint_out);

  if(!exists(checkpoint_directory))
    {
      create_directories(checkpoint_directory);
    }
  else if(!is_directory(checkpoint_directory))
    {
      throw std::runtime_error("Checkpoint directory '"
                               + checkpoint_directory.string()
                               + "'already exists, but is not a directory");
    }
  int64_t new_generation(0);
  std::set<int64_t> saved_generations;
  auto old_generation(old_generations.rbegin());
  if(old_generation != old_generations.rend())
    {
      new_generation = *old_generation + 1;
      saved_generations.insert(*old_generation);
      ++old_generation;
      // Delete all but the largest generation
      for(; old_generation != old_generations.rend(); ++old_generation)
        {
          remove(checkpoint_directory
                 / ("checkpoint_" + std::to_string(*old_generation) + "_"
                    + std::to_string(El::mpi::Rank())));
        }
    }
  boost::filesystem::path checkpoint_filename(
    checkpoint_directory
    / ("checkpoint_" + std::to_string(new_generation) + "_"
       + std::to_string(El::mpi::Rank())));

  const size_t max_retries(10);
  bool wrote_successfully(false);
  for(size_t attempt = 0; attempt < max_retries && !wrote_successfully;
      ++attempt)
    {
      boost::filesystem::ofstream checkpoint_stream(checkpoint_filename);
      if(parameters.verbosity >= Verbosity::regular && El::mpi::Rank() == 0)
        {
          std::cout << "Saving checkpoint to    : " << checkpoint_directory
                    << '\n';
        }
      // TODO: Write and read precision, num of mpi procs, and procs_per_node.
      write_local_blocks(x, checkpoint_stream);
      write_local_blocks(X, checkpoint_stream);
      write_local_blocks(y, checkpoint_stream);
      write_local_blocks(Y, checkpoint_stream);
      wrote_successfully = checkpoint_stream.good();
      if(!wrote_successfully)
        {
          if(attempt + 1 < max_retries)
            {
              std::stringstream ss;
              ss << "Error writing checkpoint file '" << checkpoint_filename
                 << "'.  Retrying " << (attempt + 2) << "/" << max_retries
                 << "\n";
              std::cerr << ss.str() << std::flush;
            }
          else
            {
              std::stringstream ss;
              ss << "Error writing checkpoint file '" << checkpoint_filename
                 << "'.  Exceeded max retries.\n";
              throw std::runtime_error(ss.str());
            }
        }
    }
  if(El::mpi::Rank() == 0)
    {
      boost::filesystem::ofstream metadata(checkpoint_directory
                                           / "checkpoint_new.json");
      metadata << "{\n    \"current\": " << new_generation
               << ",\n    \"backup\": [ ";
      if(!saved_generations.empty())
        {
          metadata << *saved_generations.begin();
        }
      metadata << " ],\n"
               << "    \"version\": \"" << SDPB_VERSION_STRING
               << "\",\n    \"options\": \n";

      boost::property_tree::write_json(metadata, to_property_tree(parameters));
      metadata << "}\n";
    }
  El::mpi::Barrier(El::mpi::COMM_WORLD);
  if(El::mpi::Rank() == 0)
    {
      rename(checkpoint_directory / "checkpoint_new.json",
             checkpoint_directory / "checkpoint.json");
    }
  
  saved_generations.insert(new_generation);
  old_generations.swap(saved_generations);
}
