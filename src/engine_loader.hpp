#ifndef __ENGINE_LOADER
#define __ENGINE_LOADER

#include <thallium.hpp>

namespace tl = thallium;

class engine_loader_t {
    static const unsigned int POOL_SIZE = 4;
protected:
    tl::engine server_engine;

    std::string server_address;
    uint16_t server_id;
    bool cuda_rdma;

public:
    engine_loader_t(const std::string& server_address, uint16_t server_id, bool cuda_rdma = false);

    const tl::engine& get_engine() const;
    uint16_t get_id() const;
    bool is_cuda_rdma_enabled() const;

    void wait_for_finalize();
};

#endif
