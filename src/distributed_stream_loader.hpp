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
    /*
    std::vector<float> buffer;
    */
    tl::bulk bulk;

    exposed_memory_t() { }

    // Move constructor
    exposed_memory_t(exposed_memory_t&& other) noexcept
        : segments(std::move(other.segments)),
          buffer(other.buffer),
          bulk(std::move(other.bulk)) {
        // Set the buffer in the source object to nullptr to avoid double deletion
        other.buffer = nullptr;
    }

    // Move assignment operator
    exposed_memory_t& operator=(exposed_memory_t&& other) noexcept {
        // Check for self-assignment
        if (this != &other) {
            // Release resources in the current object
            delete buffer;

            // Move resources from the source object
            segments = std::move(other.segments);
            buffer = other.buffer;
            bulk = std::move(other.bulk);

            // Set the buffer in the source object to nullptr to avoid double deletion
            other.buffer = nullptr;
        }
        return *this;
    }

    ~exposed_memory_t() {
        delete buffer;
    }
};


class distributed_stream_loader_t : public tl::provider<distributed_stream_loader_t> {
private:
    distributed_stream_loader_t(const engine_loader_t& _engine_loader, Task _task_type,
        unsigned int _K, unsigned int _N, unsigned int _R, unsigned int _C, int64_t seed,
        unsigned int _num_samples_per_representative, std::vector<long> _representative_shape,
        BufferStrategy _buffer_strategy,
        bool discover_endpoints = false, bool _verbose = false);
    
public:
    // This factory method and the private constructor prevent users
    // from putting an instance of distributed_stream_loader_t on the stack.
    static distributed_stream_loader_t* create(const engine_loader_t& engine_loader, Task task_type,
    unsigned int K, unsigned int N, unsigned int R, unsigned int C, int64_t seed,
    unsigned int num_samples_per_representative, std::vector<long> representative_shape,
    BufferStrategy buffer_strategy, bool discover_endpoints, bool verbose);
    ~distributed_stream_loader_t() noexcept;
    void finalize();

    void register_endpoints(const std::map<std::string, int>& endpoints);
    void start();

    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets);
    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets, std::vector<torch::Tensor> &ground_truth,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights, std::vector<torch::Tensor> &aug_ground_truth);
    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets, std::vector<torch::Tensor> &ground_truth);
    int wait();

    void use_these_allocated_variables(const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    //void use_these_allocated_variables(const torch::Tensor &aug_samples, std::vector<torch::Tensor> &aug_ground_truth, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    void enable_augmentation(bool state);
    void measure_performance(bool state);
    size_t get_rehearsal_size();
    std::vector<float> get_metrics(size_t i_batch);

protected:
    uint16_t m_provider_id;

    void init_rehearsal_buffers(bool pin_buffers);
    void init_receiving_rdma_buffer();

    void copy_last_batch(const queue_item_t &batch);
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
    cudaStream_t m_streams[3];
    /*
    std::unique_ptr<cuda::stream_t> m_client_stream_async;
    std::unique_ptr<cuda::stream_t> m_client_stream_sync;
    std::unique_ptr<cuda::stream_t> m_server_stream_sync;
    */
#endif

    bool started = false;
    bool m_augmentation_enabled = false;
    bool m_measure_performance = false;
    bool m_use_allocated_variables = false;

    torch::Tensor alloc_aug_samples;
    torch::Tensor alloc_aug_targets;
    torch::Tensor alloc_aug_weights;

    //torch::Tensor* dest_samples = nullptr;
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

    std::map<std::string, int> gather_endpoints();
    bool mpi_was_initialized = false;
    int m_rank = -1;
    int m_local_rank = -1;
    int m_num_workers = -1;

    void async_process();

    void augment_batch(queue_item_t &batch);
    std::vector<std::pair<int, int>> merge_contiguous_memory(std::vector<std::pair<int, int>>& sections) const;
    void copy_exposed_buffer_to_aug_batch(const queue_item_t &batch, const std::vector<std::pair<int, int>>& sections);
    void populate_rehearsal_buffer(const queue_item_t& batch);
    void update_representative_weights(const queue_item_t& batch, int num_representatives);

    std::unordered_map<int, std::vector<int>> pick_random_indices(int effective_representatives);
    void get_remote_samples(const tl::request& req, std::vector<tl::bulk>& client_bulks, const std::vector<int>& indices, int offset);

    std::vector<torch::Tensor*> client_dest_tensors;
    std::vector<exposed_memory_t> client_mems, server_mems;
    //exposed_memory_t client_mem, server_mem;

    std::vector<std::vector<std::tuple<int, float, size_t, size_t>>> metadata;
};

#endif
