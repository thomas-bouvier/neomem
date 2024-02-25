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
        ASSERT(targets.dim() == 1);
        ASSERT(representatives[0].sizes()[0] == targets.sizes()[0]);
        // The following assertation will fail if batches get truncated at the
        // end of the epoch, as activations and corresponding training data
        // are off sync by one iteration.
        // ASSERT(representatives.sizes()[0] == activations.sizes()[0]);
        size = representatives[0].sizes()[0];

        m_weights = torch::ones({size}, torch::TensorOptions().dtype(torch::kFloat32).device(representatives[0].device()));
    }

    queue_item_t(const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations,
            const std::vector<torch::Tensor>& aug_representatives, const torch::Tensor& aug_targets, const torch::Tensor& aug_weights, const std::vector<torch::Tensor>& buf_activations, const torch::Tensor& buf_activations_rep) :
            m_representatives(representatives), m_targets(targets), m_activations(activations),
            m_aug_representatives(aug_representatives), m_aug_targets(aug_targets), m_aug_weights(aug_weights), m_buf_activations(buf_activations), m_buf_activations_rep(buf_activations_rep)
    {
        ASSERT(targets.dim() == 1);
        ASSERT(representatives[0].sizes()[0] == targets.sizes()[0]);
        //ASSERT(representatives[0].sizes()[0] == amp.sizes()[0]);
        //ASSERT(representatives[0].sizes()[0] == ph.sizes()[0]);
        size = representatives[0].sizes()[0];
        aug_size = size;

        if (aug_representatives.size() > 0) {
            ASSERT(aug_representatives[0].dim() > 0 && aug_targets.dim() == 1);
            auto actual_R = aug_representatives[0].sizes()[0] - size;
            ASSERT(actual_R > 0 && actual_R + size == aug_targets.sizes()[0]
                                && actual_R + size == aug_weights.sizes()[0]);
                                //&& actual_R + size == aug_amp.sizes()[0]
                                //&& actual_R + size == aug_ph.sizes()[0]);
        }

        m_weights = torch::ones({size}, aug_weights.options());
    }

    queue_item_t() { }

    size_t get_size() const {
        return size;
    }
};

#endif
