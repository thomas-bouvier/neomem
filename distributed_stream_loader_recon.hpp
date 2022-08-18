#ifndef __DISTRIBUTED_STREAM_LOADER_RECON
#define __DISTRIBUTED_STREAM_LOADER_RECON

#include <torch/extension.h>
#include <thallium.hpp>
#include <unordered_map>
#include <iostream>
#include <random>

namespace tl = thallium;

typedef std::vector<torch::Tensor> diffr_representative_t;
typedef std::vector<diffr_representative_t> buffer_vecs_t;
typedef std::unordered_map<int, std::pair<double, buffer_vecs_t>> rehearsal_map_vecs_t;
typedef std::unordered_map<int, int> rehearsal_counts_t;

class distributed_stream_loader_recon_t : public tl::provider<distributed_stream_loader_recon_t> {
    const size_t MAX_QUEUE_SIZE = 1024;

    unsigned int K, N, C;
    std::default_random_engine rand_gen;
    rehearsal_map_vecs_t rehearsal_map;
    rehearsal_counts_t counts;
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
    tl::managed<tl::xstream> xstream;
    tl::managed<tl::thread> async_thread;

    std::vector<tl::provider_handle> provider_handles;
    tl::remote_procedure get_samples_procedure;

    rehearsal_map_vecs_t get_samples(const std::vector<int>& indices);
    void get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices);
    void async_process();

public:
    distributed_stream_loader_recon_t(unsigned int _N, unsigned int _C, int64_t seed,
        uint16_t server_id, const std::string& server_address,
        std::vector<std::pair<int, std::string>>& endpoints);
    ~distributed_stream_loader_recon_t();
    void add_endpoints(const std::vector<std::pair<int, std::string>>& endpoints);

    void accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
            const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights);
    int wait();
    size_t get_rehearsal_size();
    size_t get_history_count();
};

#endif
