#include <torch/extension.h>
#include <iostream>
#include <tuple>

#include "mpi.h"

#include "engine_loader.hpp"
#include "distributed_stream_loader.hpp"
#include "debug.hpp"
#define ___ASSERT
#define ___DBG

unsigned int K = 10;
unsigned int N = 10;
unsigned int R = 5;
unsigned int C = 5;
int64_t seed = 42;

int main(int argc, char** argv) {
    bool mpi = false;
    std::string server_address;
    int server_id;
    bool discover_endpoints = true;
    std::map<std::string, int> endpoints;

    // Check if --mpi option is provided
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--mpi") {
            mpi = true;
            break;
        }
    }

    if (mpi) {
        int rank, num_workers = 0;
        // MPI has maybe been initialized by horovodrun
        int mpi_initialized = true;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized)
            MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_workers);

        server_address = "na+sm";
        server_id = rank;
    } else {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <server_address> <server_id> [.. <address> <provider_id>]" << std::endl;
            exit(0);
        }

        server_address = argv[1];
        server_id = atoi(argv[2]);

        for (int i = 3; i < argc; i += 2) {
            auto endpoint = std::make_pair<>(std::string(argv[i]), atoi(argv[i + 1]));
            endpoints.insert(endpoint);
            std::cout << "Endpoint " << endpoint.first << ", " << endpoint.second << std::endl;
        }

        std::string choice;
        std::cout << "Discover endpoints via MPI? ";
        std::cin >> choice;
        if (choice == "no") {
            while (true) {
                discover_endpoints = false;
                std::string address;
                int provider_id;
                std::cout << "Endpoint to sample from (local endpoint is NOT already-included)? ";
                std::cin >> address;
                if (address == "no") break;
                std::cin >> provider_id;
                auto endpoint = std::make_pair<>(address, provider_id);
                endpoints.insert(endpoint);
                std::cout << "Endpoint " << address << ", " << provider_id << std::endl;
                std::cin.clear();
            }
        }
    }

    engine_loader_t engine(server_address, server_id);
    distributed_stream_loader_t* dsl = distributed_stream_loader_t::create(engine, REHEARSAL, K, N, C, seed, R, 1, {3, 224, 224}, 0, 0, {}, CPUBuffer, discover_endpoints, true);
    if (!mpi) {
        dsl->register_endpoints(endpoints);
    }
    dsl->enable_augmentation(true);
    dsl->start();

    MPI_Barrier(MPI_COMM_WORLD);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available on this system. Using CUDA." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "CUDA is NOT available on this system. Using CPU." << std::endl;
        device_type = torch::kCPU;
    }

    /*
    torch::Tensor aug_samples = torch::full({N + R, 3, 224, 224}, -1, torch::TensorOptions().dtype(torch::kFloat32).device(device_type));
    torch::Tensor aug_labels = torch::randint(K, {N + R}, torch::TensorOptions().dtype(torch::kInt64).device(device_type));
    torch::Tensor aug_weights = torch::zeros({N + R}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type));

    auto random_batch = [server_id, device_type](int i, torch::DeviceType d) -> std::tuple<torch::Tensor, torch::Tensor> {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(d);
        auto label = static_cast<long int>(i % K);
        torch::Tensor labels = torch::full({N}, label);
        torch::Tensor samples = torch::full({N, 3, 224, 224}, label, options);
        return std::make_tuple<>(samples, labels);
    };

    for (int i = 0; i < 100; i++) {
        std::cout << "Round " << i << std::endl;
        auto batch = random_batch(i, device_type);
        dsl->accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
        size_t size = dsl->wait();
        std::cout << "Received " << size - N << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);

        for (size_t j = 0; j < size; j++) {
            if (j < N) {
                int pixel = i % K;
                ASSERT(torch::equal(aug_samples[j], torch::full({3, 224, 224}, pixel, torch::TensorOptions().dtype(torch::kFloat32).device(device_type))));
            } else {
                ASSERT(!torch::equal(aug_samples[j], torch::full({3, 224, 224}, -1, torch::TensorOptions().dtype(torch::kFloat32).device(device_type))));
            }
            ASSERT(torch::equal(aug_samples[j][0][0][0], aug_labels[j].to(torch::kFloat32)));
        }
    }
    */

    dsl->finalize();
    engine.wait_for_finalize();

    return 0;
}
