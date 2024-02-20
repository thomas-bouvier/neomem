#ifndef __QUEUE_ITEM
#define __QUEUE_ITEM

#include "debug.hpp"

#include <torch/extension.h>

struct queue_item_t {
    torch::TensorList m_representatives;
    torch::Tensor m_targets, m_weights;
    torch::TensorList m_activations;

    // Flyweight objects.
    torch::TensorList m_aug_representatives;
    torch::Tensor m_aug_targets, m_aug_weights;

    torch::TensorList m_buf_activations;
    torch::Tensor m_buf_activations_rep;

    int size = 0;
    int aug_size = 0;
    int m_augmentation_mark = 0;

    /**
     * 
     */
    queue_item_t(const torch::TensorList& representatives, const torch::Tensor& targets, const torch::TensorList &activations) :
        m_representatives(representatives), m_targets(targets), m_activations(activations) {
            ASSERT(targets.dim() == 1);
            ASSERT(representatives[0].sizes()[0] == targets.sizes()[0]);
            // The following assertation will fail if batches get truncated at the
            // end of the epoch, as activations and corresponding training data
            // are off sync by one iteration.
            // ASSERT(representatives.sizes()[0] == activations.sizes()[0]);
            size = representatives[0].sizes()[0];

            m_weights = torch::ones({size}, torch::TensorOptions().dtype(torch::kFloat32).device(representatives[0].device()));
    }

    queue_item_t(const torch::TensorList& representatives, const torch::Tensor& targets, const torch::TensorList& activations,
            const torch::TensorList& aug_representatives, const torch::Tensor& aug_targets, const torch::Tensor& aug_weights, const torch::TensorList& buf_activations, const torch::Tensor& buf_activations_rep) :
        m_representatives(representatives), m_targets(targets), m_activations(activations),
        m_aug_representatives(aug_representatives), m_aug_targets(aug_targets), m_aug_weights(aug_weights), m_buf_activations(buf_activations), m_buf_activations_rep(buf_activations_rep) {
            ASSERT(targets.dim() == 1);
            ASSERT(representatives[0].sizes()[0] == targets.sizes()[0]);
            //ASSERT(representatives[0].sizes()[0] == amp.sizes()[0]);
            //ASSERT(representatives[0].sizes()[0] == ph.sizes()[0]);
            size = representatives[0].sizes()[0];
            aug_size = size;

            ASSERT(aug_representatives[0].dim() > 0 && aug_targets.dim() == 1);
            auto actual_R = aug_representatives[0].sizes()[0] - size;
            ASSERT(actual_R > 0 && actual_R + size == aug_targets.sizes()[0]
                                && actual_R + size == aug_weights.sizes()[0]);
                                //&& actual_R + size == aug_amp.sizes()[0]
                                //&& actual_R + size == aug_ph.sizes()[0]);

            m_weights = torch::ones({size}, aug_weights.options());
    }

    queue_item_t() { }

    size_t get_size() const {
        return size;
    }
};

#endif
