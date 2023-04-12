#ifndef __DISTRIBUTED_STREAM_LOADER
#define __DISTRIBUTED_STREAM_LOADER

#include "engine_loader.hpp"
#include "metrics.hpp"

#include <torch/extension.h>
#include <thallium.hpp>
#include <unordered_map>
#include <iostream>
#include <random>
#include <vector>

namespace tl = thallium;

enum Task { Classification, Reconstruction };
enum BufferStrategy { NoBuffer, CPUBuffer, CUDABuffer };


class distributed_stream_loader_t : public tl::provider<distributed_stream_loader_t> {
    const size_t MAX_QUEUE_SIZE = 1024;
    bool augmentation_enabled = false;

    engine_loader_t engine_loader;

    Task task_type;
    unsigned int K, N, R, C;
    std::default_random_engine rand_gen;
    unsigned int num_samples_per_representative, num_bytes_per_representative;
    std::vector<long> representative_shape;
    BufferStrategy buffer_strategy = NoBuffer;
    bool verbose;

    torch::Tensor* rehearsal_tensor = nullptr;
    std::vector<std::pair<size_t, double>> rehearsal_metadata;
    size_t history_count = 0;
    size_t rehearsal_size = 0;

    int i_batch = 0;
    std::map<int, metrics_t> metrics;

    bool started = false;

    struct queue_item_t {
        int aug_size = 0;
        torch::Tensor samples, targets, aug_samples, aug_targets, aug_weights;
        queue_item_t(const torch::Tensor &_samples, const torch::Tensor &_targets) :
            samples(_samples), targets(_targets) { }
        queue_item_t(const torch::Tensor &_samples, const torch::Tensor &_targets,
                const torch::Tensor &_aug_samples, const torch::Tensor &_aug_targets, const torch::Tensor &_aug_weights) :
            samples(_samples), targets(_targets), aug_samples(_aug_samples), aug_targets(_aug_targets), aug_weights(_aug_weights) { }
        queue_item_t() { }
    };

    struct exposed_memory_t {
        std::vector<std::pair<void*, std::size_t>> segments;
        torch::Tensor* buffer = nullptr;
        tl::bulk bulk;

        exposed_memory_t() { }
    };

    bool use_allocated_variables = false;
    torch::Tensor alloc_aug_samples;
    torch::Tensor alloc_aug_targets;
    torch::Tensor alloc_aug_weights;

    std::deque<queue_item_t> request_queue, response_queue;
    tl::mutex request_mutex;
    tl::condition_variable request_cond;
    tl::mutex rehearsal_mutex;

    // client thread
    tl::managed<tl::xstream> es;
    tl::managed<tl::thread> async_thread;
    std::vector<tl::provider_handle> provider_handles;
    tl::remote_procedure get_samples_procedure;

    std::map<std::string, int> gather_endpoints() const;
    void init_rehearsal_buffers(bool pin_buffers);

    void async_process();

    void augment_batch(queue_item_t &batch, int batch_size);
    void copy_exposed_buffer_to_aug_batch(queue_item_t &batch, int batch_size);
    void populate_rehearsal_buffer(const queue_item_t& batch);
    void update_representative_weights(int num_representatives, int batch_size);

    std::unordered_map<int, std::vector<int>> pick_random_indices(int effective_representatives);
    void get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices, int offset);

    exposed_memory_t client_mem, server_mem;

public:
    distributed_stream_loader_t(const engine_loader_t& engine, Task _task_type,
        unsigned int _K, unsigned int _N, unsigned int _R, unsigned int _C, int64_t seed,
        unsigned int _num_samples_per_representative, std::vector<long> _representative_shape,
        BufferStrategy _buffer_strategy,
        bool discover_endpoints = false, bool _verbose = false);
    ~distributed_stream_loader_t();

    void register_endpoints(const std::map<std::string, int>& endpoints);
    void start();

    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets);
    int wait();

    void use_these_allocated_variables(const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    void expose_memory(exposed_memory_t &mem);
    void copy_last_batch(queue_item_t &batch, int batch_size);

    void enable_augmentation(bool state);
    size_t get_rehearsal_size();
    size_t get_history_count();
    std::vector<float> get_metrics(size_t i_batch);
};

#endif
