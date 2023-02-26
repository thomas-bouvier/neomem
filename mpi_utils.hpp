#include <mpi.h>
#include <iostream>
#include <map>
#include <string>

using dictionary = std::map<std::string, int>;

dictionary gather_dictionary(const dictionary &dict, int num_workers) {
    int num_elements = dict.size();

    // Determine the sizes of the local strings
    std::vector<int> string_sizes(num_elements);
    int i = 0;
    for (const auto& pair : dict) {
        string_sizes[i] = pair.first.size();
        i++;
    }

    // Allgather the string sizes to determine the total size of the allgathered strings
    std::vector<int> all_sizes(num_elements * num_workers);
    MPI_Allgather(string_sizes.data(), num_elements, MPI_INT, all_sizes.data(), num_elements, MPI_INT, MPI_COMM_WORLD);
    int keys_total_size = 0;
    for (int i = 0; i < num_elements * num_workers; i++) {
        keys_total_size += all_sizes[i];
    }

    // Create a buffer to hold the allgathered strings
    std::vector<char> gathered_keys(keys_total_size);
    std::vector<int> gathered_values(num_elements * num_workers);

    // Determine the displacements for the allgatherv operation
    std::vector<int> displacements(num_elements * num_workers, 0);
    for (int i = 1; i < num_elements * num_workers; i++) {
        displacements[i] = displacements[i - 1] + all_sizes[i - 1];
    }

    // Copy the keys and values from the map into the arrays
    std::string local_keys;
    std::vector<int> local_values;
    //int current_offset = 0;
    for (auto& [key, value] : dict) {
        /*
        std::copy(key.begin(), key.end(), gathered_keys.begin() + current_offset);
        current_offset += key.size();
        */
        local_keys += key;
        local_values.push_back(value);
    }

    MPI_Allgatherv(local_keys.data(), local_keys.size(), MPI_CHAR, gathered_keys.data(), all_sizes.data(), displacements.data(), MPI_CHAR, MPI_COMM_WORLD);
    MPI_Allgather(local_values.data(), num_elements, MPI_INT, gathered_values.data(), num_elements, MPI_INT, MPI_COMM_WORLD);

    // Construct the final map
    std::map<std::string, int> gathered_map;
    int string_start = 0;
    for (int i = 0; i < num_workers * num_elements; i++) {
        int string_length = all_sizes[i];
        std::string str(gathered_keys.begin() + string_start, gathered_keys.begin() + string_start + string_length);
        gathered_map[str] = gathered_values[i];
        string_start += string_length;
    }

    return gathered_map;
}
