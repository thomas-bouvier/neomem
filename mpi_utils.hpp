#include <mpi.h>
#include <iostream>
#include <map>
#include <string>
#include <cstring>

#define MAX_CF_LENGTH 55

using dictionary = std::map<std::string, int>;

dictionary gather_dictionary(dictionary &dict, int max_key_length, int num_workers, int rank) {
    // Calculate destination dictionary size
    int num_keys = dict.size();
    int total_length = num_keys * max_key_length;
    int final_num_keys = 0;
    MPI_Allreduce(&num_keys, &final_num_keys, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Computing number of elements that are received from each process
    int *recvcounts = NULL;
    recvcounts = new int[num_workers];
    MPI_Allgather(&total_length, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Computing displacement relative to recvbuf at which to place the incoming data from each process
    int *displs = NULL;
    int totLen = 0;
    displs = new int[num_workers];
    displs[0] = 0;
    totLen += recvcounts[0] + 1;
    for (int i = 1; i < num_workers; i++) {
        totLen += recvcounts[i];
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    char(*dict_keys)[max_key_length];
    char(*final_dict_keys)[max_key_length];
    dict_keys = (char(*)[max_key_length]) malloc(num_keys * sizeof(*dict_keys));
    final_dict_keys = (char(*)[max_key_length]) malloc(final_num_keys * sizeof(*final_dict_keys));

    // Collect keys for each process
    int i = 0;
    for (auto pair : dict) {
        strncpy(dict_keys[i], pair.first.c_str(), max_key_length);
        i++;
    }
    MPI_Allgatherv(dict_keys, total_length, MPI_CHAR, final_dict_keys, recvcounts, displs, MPI_CHAR, MPI_COMM_WORLD);

    // Create new dictionary and distribute it to all processes
    dict.clear();
    for (int i = 0; i < final_num_keys; i++) {
        dict[final_dict_keys[i]] = dict.size();
    }

    delete[] dict_keys;
    delete[] final_dict_keys;
    delete[] recvcounts;
    delete[] displs;

    return dict;
}
