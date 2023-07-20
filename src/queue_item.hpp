#ifndef __QUEUE_ITEM
#define __QUEUE_ITEM

#include "debug.hpp"

#include <torch/extension.h>

struct queue_item_t {
    torch::Tensor samples, targets, weights, aug_samples, aug_targets, aug_weights;

    int size = 0;
    int aug_size = 0;
    int m_augmentation_mark = 0;

    queue_item_t(const torch::Tensor &_samples, const torch::Tensor &_targets) :
        samples(_samples), targets(_targets) {
            ASSERT(targets.dim() == 1 && samples.sizes()[0] == targets.sizes()[0]);
            size = samples.sizes()[0];

            weights = torch::ones({size}, targets.options());
        }

    queue_item_t(const torch::Tensor &_samples, const torch::Tensor &_targets,
            const torch::Tensor &_aug_samples, const torch::Tensor &_aug_targets, const torch::Tensor &_aug_weights) :
        samples(_samples), targets(_targets), aug_samples(_aug_samples), aug_targets(_aug_targets), aug_weights(_aug_weights) {
            ASSERT(targets.dim() == 1 && samples.sizes()[0] == targets.sizes()[0]);
            size = samples.sizes()[0];

            ASSERT(aug_samples.dim() > 0 && aug_targets.dim() == 1);
            auto actual_R = aug_samples.sizes()[0] - size;
            ASSERT(actual_R > 0 && actual_R + size == aug_targets.sizes()[0]
                && actual_R + size == aug_weights.sizes()[0]);

            weights = torch::ones({size}, targets.options());
        }

    queue_item_t() { }

    size_t get_size() const {
        return size;
    }
};

#endif