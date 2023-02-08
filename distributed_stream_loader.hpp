#ifndef __DISTRIBUTED_STREAM_LOADER
#define __DISTRIBUTED_STREAM_LOADER

#include <torch/extension.h>
#include <thallium.hpp>
#include <unordered_map>
#include <iostream>
#include <random>
#include <vector>

namespace tl = thallium;

enum Task { Classification, Reconstruction };

typedef std::vector<torch::Tensor> representative_t;
typedef std::vector<representative_t> buffer_t;
typedef std::unordered_map<int, std::pair<double, buffer_t>> rehearsal_map_t;

class engine_loader_t {
    static const unsigned int POOL_SIZE = 4;
protected:
    tl::engine server_engine;
    uint16_t server_id;

public:
    engine_loader_t(const std::string& server_address, uint16_t server_id);
    ~engine_loader_t();
};


class distributed_stream_loader_t : public engine_loader_t, public tl::provider<distributed_stream_loader_t> {
    const size_t MAX_QUEUE_SIZE = 1024;

    Task task_type;
    unsigned int K, N, C;
    std::default_random_engine rand_gen;
    unsigned int num_samples_per_representative;
    std::vector<long> representative_shape;
    bool cuda_rdma, verbose;

    std::vector<torch::Tensor> rehearsal_vector;
    std::vector<std::pair<size_t, double>> rehearsal_metadata;
    size_t history_count = 0;
    size_t rehearsal_size = 0;

    struct queue_item_t {
        int aug_size = 0;
        torch::Tensor samples, targets, aug_samples, aug_targets, aug_weights;
        queue_item_t(const torch::Tensor &_samples, const torch::Tensor &_targets,
                const torch::Tensor &_aug_samples, const torch::Tensor &_aug_targets, const torch::Tensor &_aug_weights) :
            samples(_samples), targets(_targets), aug_samples(_aug_samples), aug_targets(_aug_targets), aug_weights(_aug_weights) { }
        queue_item_t() { }
    };

    std::deque<queue_item_t> request_queue, response_queue;
    tl::mutex request_mutex;
    tl::condition_variable request_cond;
    tl::mutex rehearsal_mutex;

    // client thread
    tl::managed<tl::xstream> es;
    tl::managed<tl::thread> async_thread;

    std::vector<tl::provider_handle> provider_handles;
    tl::remote_procedure get_samples_procedure;

    std::unordered_map<int, std::vector<int>> pick_random_indices(int effective_representatives);
    rehearsal_map_t get_samples(const std::vector<int>& indices);
    void get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices);
    void populate_rehearsal_buffer(const queue_item_t& batch, int batch_size);
    void update_representative_weights(int effective_representatives, int batch_size);
    std::map<std::string, int> gather_endpoints() const;

public:
    distributed_stream_loader_t(Task _task_type, unsigned int _K, unsigned int _N, unsigned int _C, int64_t seed,
        uint16_t server_id, const std::string& server_address,
        unsigned int _num_samples_per_representative, std::vector<long> _representative_shape,
        bool _cuda_rdma, bool discover_endpoints = false,
        bool _verbose = false);
    ~distributed_stream_loader_t();

    void register_endpoints(const std::map<std::string, int>& endpoints);
    void async_process();

    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    int wait();
    size_t get_rehearsal_size();
    size_t get_history_count();
};

#endif
