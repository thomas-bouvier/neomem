#include <torch/extension.h>
#include <iostream>
#include <tuple>

#include "distributed_stream_loader.hpp"

unsigned int K = 10;
unsigned int N = 20;
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
    while (true) {
        std::string address;
        int provider_id;
        std::cout << "Endpoint to sample from (local endpoint is not already-included)? ";
        std::cin >> address;
        if (address == "no") break;
        std::cin >> provider_id;
        auto endpoint = std::make_pair<>(address, provider_id);
        endpoints.insert(endpoint);
        std::cout << "Endpoint " << address << ", " << provider_id << std::endl;
        std::cin.clear();
    }

    distributed_stream_loader_t dsl(Classification, K, N, C, seed, server_id, server_address, 2, {3, 224, 224}, false);
    std::cout << "size " << endpoints.size() << std::endl;
    for (auto endpoint : endpoints) {
        std::cout << "Checking " << endpoint.first << ", " << endpoint.second << std::endl;
    }
    dsl.register_endpoints(endpoints);

    torch::Tensor aug_samples = torch::zeros({N + R, 3, 224, 224});
    torch::Tensor aug_labels = torch::randint(K, {N + R});
    torch::Tensor aug_weights = torch::zeros({N + R});

    auto random_batch = [server_id]() -> std::tuple<torch::Tensor, torch::Tensor> {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor labels = torch::randint(K, {N});
        torch::Tensor samples = torch::full({N, 3, 224, 224}, server_id, options);
        return std::make_tuple<>(samples, labels);
    };

    std::cout << "Round 1" << std::endl;
    auto batch = random_batch();
    dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
    int size = dsl.wait();
    std::cout << "Received " << size - N << std::endl;

    std::cout << "Round 2" << std::endl;
    batch = random_batch();
    dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
    size = dsl.wait();
    std::cout << "Received " << size - N << std::endl;

    for (int i = 3; i < 1000; i++) {
        std::cout << "Round " << i << std::endl;
        batch = random_batch();
        dsl.accumulate(std::get<0>(batch), std::get<1>(batch), aug_samples, aug_labels, aug_weights);
        size = dsl.wait();
        std::cout << "Received " << size - N << std::endl;
    }

    return 0;
}
