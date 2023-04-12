#include <torch/extension.h>
#include <iostream>
#include <tuple>

#include "distributed_stream_loader.hpp"
#include "debug.hpp"
#define ___ASSERT

unsigned int K = 10;
unsigned int N = 10;
unsigned int R = 5;
unsigned int C = 5;
int64_t seed = 42;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <server_address> <server_id> [.. <address> <provider_id>]" << std::endl;
        exit(0);
    }

    std::string server_address = argv[1];
    uint16_t server_id = atoi(argv[2]);

    std::map<std::string, int> endpoints;
    for (int i = 3; i < argc; i += 2) {
        auto endpoint = std::make_pair<>(std::string(argv[i]), atoi(argv[i + 1]));
        endpoints.insert(endpoint);
        std::cout << "Endpoint " << endpoint.first << ", " << endpoint.second << std::endl;
    }

    bool discover_endpoints = true;
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

    engine_loader_t engine(server_address, server_id);
    distributed_stream_loader_t dsl(engine, Classification, K, N, R, C, seed, 1, {3, 224, 224}, discover_endpoints, true);
    dsl.register_endpoints(endpoints);
    dsl.enable_augmentation(true);
    dsl.start();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor aug_samples = torch::full({N + R, 3, 224, 224}, -1, options);
    torch::Tensor aug_labels = torch::randint(K, {N + R}, options);
    torch::Tensor aug_weights = torch::zeros({N + R}, options);

    auto random_batch = [server_id](int i) -> std::tuple<torch::Tensor, torch::Tensor> {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto label = static_cast<double>(i % K);
        torch::Tensor labels = torch::full({N}, label);
        torch::Tensor samples = torch::full({N, 3, 224, 224}, label, options);
        return std::make_tuple<>(samples, labels);
    };

    for (int i = 0; i < 1000; i++) {
        std::cout << "Round " << i << std::endl;
        auto batch = random_batch(i);
        dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
        size_t size = dsl.wait();
        std::cout << "Received " << size - N << std::endl;

        for (size_t j = 0; j < size; j++) {
            if (j < N) {
                int pixel = i % K;
                ASSERT(torch::equal(aug_samples[j], torch::full({3, 224, 224}, pixel, options)));
            } else {
                ASSERT(!torch::equal(aug_samples[j], torch::full({3, 224, 224}, -1, options)));
            }
        }
    }

    return 0;
}
