#include "engine_loader.hpp"
#include "debug.hpp"

engine_loader_t::engine_loader_t(const std::string &address, uint16_t provider_id, bool _cuda_rdma)
    : server_address(address), server_id(provider_id), cuda_rdma(_cuda_rdma) {
    struct hg_init_info hii;
    memset(&hii, 0, sizeof(hii));
    if (cuda_rdma) {
        hii.na_init_info.request_mem_device = true;
    }

    // Use a progress thread to avoid collisions (hangs) with MPI collectives
    server_engine = tl::engine(address, THALLIUM_SERVER_MODE, true, POOL_SIZE, &hii);
    server_engine.enable_remote_shutdown();

    std::cout << "Server running at address " << server_engine.self()
                << " with provider id " << provider_id << ", device registration: " << cuda_rdma << std::endl;
}

void engine_loader_t::wait_for_finalize() {
    server_engine.wait_for_finalize();
}

const tl::engine& engine_loader_t::get_engine() const {
    return server_engine;
}

uint16_t engine_loader_t::get_id() const {
    return server_id;
}

bool engine_loader_t::is_cuda_rdma_enabled() const {
    return cuda_rdma;
}
