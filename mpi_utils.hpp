#include <mpi.h>
#include <iostream>
#include <map>
#include <string>

using dictionary = std::map<std::string, int>;

dictionary gather_dictionary(const dictionary &dict, int num_workers) {
    int map_size = dict.size();
    MPI_Allgather(&map_size, 1, MPI_INT, &map_size, 1, MPI_INT, MPI_COMM_WORLD);

    // Define the arrays to hold the map data
    std::string* keys = new std::string[map_size];
    int* values = new int[map_size];

    // Copy the keys and values from the map into the arrays
    int i = 0;
    for (auto& [key, value] : dict) {
        keys[i] = key;
        values[i] = value;
        i++;
    }

    // Exchange the keys and values arrays between all processes
    MPI_Allgather(keys, map_size, MPI_CHAR, keys, map_size, MPI_CHAR, MPI_COMM_WORLD);
    MPI_Allgather(values, map_size, MPI_INT, values, map_size, MPI_INT, MPI_COMM_WORLD);

    // Construct the final map
    std::map<std::string, int> gathered_map;
    for (int i = 0; i < num_workers * map_size; i++) {
        gathered_map[keys[i]] = values[i];
    }

    // Clean up
    delete[] keys;
    delete[] values;

    return gathered_map;
}
