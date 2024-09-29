#ifndef __MEMORY_UTILS_HPP
#define __MEMORY_UTILS_HPP

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef WITHOUT_CUDA
#include <cuda_runtime.h>
#endif

// NullStream struct for non-CUDA path
struct NullStream {};

// Template declaration for copy_memory
template<typename StreamType>
void copy_memory(void* dst, const void* src, size_t size, StreamType stream);

// Specialization for non-CUDA path
template<>
inline void copy_memory<NullStream>(void* dst, const void* src, size_t size, NullStream) {
    std::memcpy(dst, src, size);
}

#ifndef WITHOUT_CUDA
// Specialization for CUDA path
template<>
inline void copy_memory<cudaStream_t>(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
}
#endif

// smart_copy function
#ifndef WITHOUT_CUDA
inline void smart_copy(void* dst, const void* src, size_t size, cudaStream_t stream) {
    copy_memory(dst, src, size, stream);
}
#else
inline void smart_copy(void* dst, const void* src, size_t size, NullStream = NullStream{}) {
    copy_memory(dst, src, size, NullStream{});
}
#endif

std::vector<std::pair<int, int>> merge_contiguous_memory(std::vector<std::pair<int, int>>& sections);

#endif // __MEMORY_UTILS_HPP
