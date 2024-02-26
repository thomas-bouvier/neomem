#ifndef __EXPOSED_MEMORY
#define __EXPOSED_MEMORY

#include <torch/extension.h>

struct exposed_memory_attr {
    bool cuda;
    tl::bulk_mode bulk_mode;
};

struct exposed_memory_t {
    std::vector<std::pair<void*, std::size_t>> segments;
    std::unique_ptr<torch::Tensor> buffer;
    tl::bulk bulk;

    exposed_memory_t() = default;

    // Move constructor
    exposed_memory_t(exposed_memory_t&& other) noexcept
        : segments(std::move(other.segments)),
          buffer(std::move(other.buffer)),
          bulk(std::move(other.bulk)) { }

    // Move assignment operator
    exposed_memory_t& operator=(exposed_memory_t&& other) noexcept {
        // Check for self-assignment
        if (this != &other) {
            // Move resources from the source object
            segments = std::move(other.segments);
            buffer = std::move(other.buffer);
            bulk = std::move(other.bulk);

            // Set the buffer in the source object to nullptr to avoid double deletion
            other.buffer = nullptr;
        }
        return *this;
    }
};

#endif
