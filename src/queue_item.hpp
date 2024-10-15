#ifndef __QUEUE_ITEM
#define __QUEUE_ITEM

#include "debug.hpp"

#include <torch/extension.h>

struct queue_item_t {
    std::vector<torch::Tensor> m_representatives;
    torch::Tensor m_targets, m_weights;
    std::vector<torch::Tensor> m_activations;

    // Flyweight objects.
    std::vector<torch::Tensor> m_aug_representatives;
    torch::Tensor m_aug_targets, m_aug_weights;

    std::vector<torch::Tensor> m_buf_activations;
    torch::Tensor m_buf_activations_rep;

    int size = 0;
    int aug_size = 0;
    int m_augmentation_mark = 0;
    int activations_size = 0;

    /**
     * 
     */
    queue_item_t(const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations) :
            m_representatives(representatives), m_targets(targets), m_activations(activations)
    {
        ASSERT(m_targets.dim() == 1);
        ASSERT(m_representatives[0].sizes()[0] == m_targets.sizes()[0]);
        //if (m_activations.size() > 0) {
        //    ASSERT(m_representatives.at(0).sizes()[0] == m_activations.at(0).sizes()[0])
        //}
        size = m_representatives.at(0).sizes()[0];

        m_weights = torch::ones({size}, torch::TensorOptions().dtype(torch::kFloat32).device(m_representatives[0].device()));
    }

    queue_item_t(const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations,
            const std::vector<torch::Tensor>& aug_representatives, const torch::Tensor& aug_targets, const torch::Tensor& aug_weights, const std::vector<torch::Tensor>& buf_activations, const torch::Tensor& buf_activations_rep) :
            m_representatives(representatives), m_targets(targets), m_activations(activations),
            m_aug_representatives(aug_representatives), m_aug_targets(aug_targets), m_aug_weights(aug_weights), m_buf_activations(buf_activations), m_buf_activations_rep(buf_activations_rep)
    {
        ASSERT(m_targets.dim() == 1);
        ASSERT(m_representatives.at(0).sizes()[0] == m_targets.sizes()[0]);
        //if (m_activations.size() > 0) {
        //    ASSERT(m_representatives.at(0).sizes()[0] == m_activations.at(0).sizes()[0])
        //}
        size = m_representatives.at(0).sizes()[0];
        aug_size = size;

        if (m_aug_representatives.size() > 0) {
            ASSERT(m_aug_representatives.at(0).dim() > 0 && m_aug_targets.dim() == 1);
            auto actual_R = m_aug_representatives.at(0).sizes()[0] - size;
            ASSERT(actual_R > 0 && actual_R + size == m_aug_targets.sizes()[0]
                                && actual_R + size == m_aug_weights.sizes()[0]);
        }

        m_weights = torch::ones({size}, m_aug_weights.options());
    }

    queue_item_t() = default;

    size_t get_size() const {
        return size;
    }
};

#endif
