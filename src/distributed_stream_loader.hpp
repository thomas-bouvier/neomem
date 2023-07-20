#ifndef __DISTRIBUTED_STREAM_LOADER
#define __DISTRIBUTED_STREAM_LOADER

#include "engine_loader.hpp"
#include "queue_item.hpp"
#include "metrics.hpp"

#include <torch/extension.h>
#include <thallium.hpp>
#include <unordered_map>
#include <iostream>
#include <random>
#include <vector>

#ifndef WITHOUT_CUDA
#include "../third_party/cuda-api-wrappers/src/cuda/api.hpp"
#endif

namespace tl = thallium;

enum Task { Classification, Reconstruction };
enum BufferStrategy { NoBuffer, CPUBuffer, CUDABuffer };

struct exposed_memory_t {
    std::vector<std::pair<void*, std::size_t>> segments;
    torch::Tensor* buffer = nullptr;
    tl::bulk bulk;

    exposed_memory_t() { }
    ~exposed_memory_t() {
        delete buffer;
    }
};


class distributed_stream_loader_t : public tl::provider<distributed_stream_loader_t> {
public:
    distributed_stream_loader_t(const engine_loader_t& _engine_loader, Task _task_type,
        unsigned int _K, unsigned int _N, unsigned int _R, unsigned int _C, int64_t seed,
        unsigned int _num_samples_per_representative, std::vector<long> _representative_shape,
        BufferStrategy _buffer_strategy,
        bool discover_endpoints = false, bool _verbose = false);
    ~distributed_stream_loader_t() noexcept;

    void register_endpoints(const std::map<std::string, int>& endpoints);
    void start();

    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets);
    int wait();

    void use_these_allocated_variables(const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    void enable_augmentation(bool state);
    void measure_performance(bool state);
    size_t get_rehearsal_size();
    std::vector<float> get_metrics(size_t i_batch);

protected:
    engine_loader_t engine_loader;

    void init_receiving_rdma_buffer(exposed_memory_t &mem);
    void copy_last_batch(queue_item_t &batch);
    std::size_t dispatch_rpcs(std::vector<tl::async_response> &responses);
    void resolve_rpcs(std::vector<tl::async_response> &responses, queue_item_t &batch);

    const size_t MAX_QUEUE_SIZE = 1024;

    //---------------------------------Will be moved to buffer class
    Task task_type;
    unsigned int K, N, R, C;
    std::default_random_engine rand_gen;
    unsigned int num_samples_per_representative, num_bytes_per_representative;
    std::vector<long> representative_shape;
    BufferStrategy buffer_strategy = NoBuffer;
    bool verbose;

    torch::Tensor* rehearsal_tensor = nullptr;
    std::vector<std::pair<size_t, float>> rehearsal_metadata;
    std::vector<int> rehearsal_counts;
    size_t m_rehearsal_size = 0;
    //---------------------------------Will be moved to buffer class

    int i_batch = 0;
    std::map<int, metrics_t> m_metrics;

#ifndef WITHOUT_CUDA
    cuda::stream_t m_client_stream_async = cuda::device::current::get().create_stream(cuda::stream::async);
    cuda::stream_t m_client_stream_sync = cuda::device::current::get().create_stream(cuda::stream::sync);
    cuda::stream_t m_server_stream_sync = cuda::device::current::get().create_stream(cuda::stream::sync);
#endif

    bool started = false;
    bool m_augmentation_enabled = false;
    bool m_measure_performance = false;
    bool m_use_allocated_variables = false;

    torch::Tensor alloc_aug_samples;
    torch::Tensor alloc_aug_targets;
    torch::Tensor alloc_aug_weights;

    torch::Tensor* dest_samples = nullptr;
    torch::Tensor* dest_targets = nullptr;
    torch::Tensor* dest_weights = nullptr;

    std::deque<queue_item_t> request_queue, response_queue;
    tl::mutex request_mutex;
    tl::condition_variable request_cond;
    tl::mutex rehearsal_mutex;

    // client thread
    tl::managed<tl::xstream> es;
    tl::managed<tl::thread> async_thread;
    std::vector<tl::provider_handle> provider_handles;
    tl::remote_procedure m_client_procedure, m_server_procedure;

    std::map<std::string, int> gather_endpoints() const;
    void init_rehearsal_buffers(bool pin_buffers);

    void async_process();

    void augment_batch(queue_item_t &batch);
    void copy_exposed_buffer_to_aug_batch(queue_item_t &batch);
    void populate_rehearsal_buffer(const queue_item_t& batch);
    void update_representative_weights(const queue_item_t& batch, int num_representatives);

    std::unordered_map<int, std::vector<int>> pick_random_indices(int effective_representatives);
    void get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices, int offset);

    exposed_memory_t client_mem, server_mem;

    std::vector<std::vector<std::tuple<int, float, size_t>>> metadata;
};

#endif
