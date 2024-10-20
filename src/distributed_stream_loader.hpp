#ifndef __DISTRIBUTED_STREAM_LOADER
#define __DISTRIBUTED_STREAM_LOADER

#include "engine_loader.hpp"
#include "exposed_memory.hpp"
#include "debug.hpp"
#include "queue_item.hpp"
#include "metrics.hpp"
#include "random_buffer.hpp"
#include "rpc_response.hpp"

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

enum BufferStrategy { NoBuffer, CPUBuffer, CUDABuffer };

struct Config {
    unsigned int C;
    unsigned int R;
    unsigned int R_distillation;
    BufferStrategy buffer_strategy;
    bool discover_endpoints;
    bool global_sampling;
    bool verbose;
    int64_t seed;
    RehearsalConfig rehearsal_config;
};

class distributed_stream_loader_t : public tl::provider<distributed_stream_loader_t> {
private:
    distributed_stream_loader_t(const engine_loader_t& _engine_loader, const Config& config);
    
public:
    static distributed_stream_loader_t* create(const engine_loader_t& engine_loader, Task task_type,
        unsigned int K, unsigned int N, unsigned int C, int64_t seed,
        unsigned int R, unsigned int num_samples_per_representative, std::vector<long> representative_shape,
        unsigned int R_distillation, unsigned int num_samples_per_activation, std::vector<long> activation_shape,
        BufferStrategy buffer_strategy, bool discover_endpoints, bool global_sampling, bool half_precision, bool verbose);
    ~distributed_stream_loader_t() noexcept;
    void finalize();

    void register_endpoints(const std::map<std::string, int>& endpoints);
    void start();

    void accumulate(const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations);
    void accumulate(const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations,
        const std::vector<torch::Tensor>& aug_representatives, const torch::Tensor& aug_targets, const torch::Tensor& aug_weights,
        const std::vector<torch::Tensor>& buf_activations, const torch::Tensor& buf_activations_rep
    );

    std::tuple<int, int> wait();

    void use_these_allocated_variables(
        const std::vector<torch::Tensor>& buf_representatives, const torch::Tensor& buf_targets, const torch::Tensor& buf_weights,
        const std::vector<torch::Tensor>& buf_activations, const torch::Tensor& buf_activations_rep
    );

    void enable_augmentation(bool state);
    void measure_performance(bool state);
    size_t get_rehearsal_size();
    std::vector<float> get_metrics(size_t i_batch);

protected:
    uint16_t m_provider_id;
    Config m_config;
    RandomBuffer m_buffer;

    void register_procedures();
    void initialize_mpi();
    void initialize_cuda();

    void initialize_rdma_buffers();
    void init_receiving_rdma_buffer(std::vector<exposed_memory_t>& server_mems, std::vector<exposed_memory_t>& client_mems, size_t nelements, size_t nsamples_per_element, std::vector<long> sample_shape);

    void copy_last_batch(const queue_item_t &batch);
    void dispatch_rpcs(std::vector<tl::async_response> &responses);
    void resolve_rpcs(std::vector<tl::async_response> &responses, queue_item_t &batch);

    const size_t MAX_QUEUE_SIZE = 1024;
    std::default_random_engine m_rand_gen;
    int i_batch = 0;
    std::map<int, metrics_t> m_metrics;

#ifndef WITHOUT_CUDA
    std::array<cudaStream_t, 4> m_streams;
#endif

    bool started = false;
    bool m_augmentation_enabled = false;
    bool m_measure_performance = false;
    bool m_use_allocated_variables = false;

    std::shared_ptr<std::vector<torch::Tensor>> m_buf_representatives;
    std::shared_ptr<torch::Tensor> m_buf_targets, m_buf_weights;
    std::shared_ptr<std::vector<torch::Tensor>> m_buf_activations;
    std::shared_ptr<torch::Tensor> m_buf_activations_rep;

    std::vector<exposed_memory_t> m_client_mems, m_server_mems;
    std::vector<exposed_memory_t> m_client_activations_mem, m_server_activations_mem;
    std::vector<exposed_memory_t> m_client_activations_rep_mem, m_server_activations_rep_mem;

    std::deque<queue_item_t> request_queue, response_queue;
    tl::mutex request_mutex;
    tl::condition_variable request_cond;
    tl::mutex rehearsal_mutex;

    // client thread
    tl::managed<tl::xstream> es;
    tl::managed<tl::thread> async_thread;
    std::vector<tl::provider_handle> provider_handles;
    tl::remote_procedure m_server_representatives_procedure, m_server_activations_procedure;
    tl::remote_procedure m_client_representatives_procedure, m_client_activations_procedure;

    std::map<std::string, int> gather_endpoints();
    bool mpi_was_initialized = false;
    int m_rank = -1;
    int m_local_rank = -1;
    int m_num_workers = -1;

    void async_process();

    void augment_batch(queue_item_t &batch);
    void copy_exposed_buffer_to_python_batch(const queue_item_t &batch, const std::vector<std::pair<int, int>>& sections);
    void create_exposed_memory(std::vector<exposed_memory_t>& memory, size_t nelements, size_t nsamples_per_element, std::vector<long> sample_shape, exposed_memory_attr attr);

    std::unordered_map<int, std::vector<int>> pick_random_indices(int effective_representatives, bool global_sampling);

    int count_samples(const std::vector<std::tuple<size_t, float, std::vector<int>>>& samples) const;
    void get_remote_representatives(
        const tl::request& req,
        std::vector<tl::bulk>& client_bulks, const std::vector<int>& indices, int offset
    );
    void get_remote_activations(
        const tl::request& req,
        std::vector<tl::bulk>& client_activations_bulk, std::vector<tl::bulk>& client_activations_rep_bulk,
        const std::vector<int>& indices, int offset
    );

    std::vector<std::vector<rpc_response_t>> metadata;
};

#endif
