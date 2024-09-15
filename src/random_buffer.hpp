#ifndef __RANDOM_BUFFER_HPP
#define __RANDOM_BUFFER_HPP

#include "queue_item.hpp"

#include <torch/extension.h>
#include <vector>
#include <random>
#include <memory>
#include <thallium.hpp>

namespace tl = thallium;

enum Task { 
    REHEARSAL,
    KD,
    REHEARSAL_KD
};

struct RehearsalConfig {
    Task task_type;
    unsigned int K;
    unsigned int N;
    unsigned int num_samples_per_representative;
    std::vector<long> representative_shape;
    unsigned int num_samples_per_activation;
    std::vector<long> activation_shape;
    bool half_precision;
    int64_t seed;
};

class RandomBuffer {
public:
    RandomBuffer(RehearsalConfig config);
    void allocate(std::unique_ptr<torch::Tensor>& storage, size_t nsamples, std::vector<long> sample_shape, bool pin_buffers);

    RehearsalConfig m_config;
    std::unique_ptr<torch::Tensor> m_rehearsal_representatives;
    std::unique_ptr<torch::Tensor> m_rehearsal_activations;
    unsigned int m_num_bytes_per_representative, m_num_bytes_per_activation;

    void populate(const queue_item_t& batch, unsigned int nelements);
    std::vector<std::tuple<size_t, float, std::vector<int>>> get_indices(const std::vector<int>& indices) const;
    size_t get_rehearsal_size();

protected:
    std::vector<std::pair<size_t, float>> rehearsal_metadata;
    std::vector<int> rehearsal_counts;
    size_t m_rehearsal_size = 0;
    std::default_random_engine m_rand_gen;

    void initialize_num_bytes_per_representative();
    void initialize_num_bytes_per_activation();
    void update_representative_weights(const queue_item_t& batch, int num_representatives);
};

#endif // __RANDOM_BUFFER_HPP
