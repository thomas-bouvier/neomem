#ifndef __EXPOSED_MEMORY
#define __EXPOSED_MEMORY

#include <torch/extension.h>

struct exposed_memory_t {
    std::vector<std::pair<void*, std::size_t>> segments;
    torch::Tensor* buffer = nullptr;
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

#endif
