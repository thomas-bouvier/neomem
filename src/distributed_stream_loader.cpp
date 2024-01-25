#include "distributed_stream_loader.hpp"
#include "mpi_utils.hpp"
#include "timer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <cereal/types/string.hpp>
#include <nvtx3/nvtx3.hpp>
#include <thallium/serialization/stl/tuple.hpp>
#include <thallium/serialization/stl/vector.hpp>


using namespace torch::indexing;

#ifndef WITHOUT_CUDA

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#endif


/**
 * This constructor initializes the provider. This class is both a server and a
 * client. There are n clients and n servers. Each client can get data from the
 * n servers. (n x n relation).
 *
 * A client holds some data in a "rehearsal buffer". A client updates the
 * content of its rehearsal buffer by sampling from the n other rehearsal
 * buffers.
 */
distributed_stream_loader_t::distributed_stream_loader_t(const engine_loader_t& _engine_loader, Task _task_type,
    unsigned int _K, unsigned int _N, unsigned int _R, unsigned int _C, int64_t seed,
    unsigned int _num_samples_per_representative, std::vector<long> _representative_shape,
    BufferStrategy _buffer_strategy, bool discover_endpoints, bool _verbose)
        : tl::provider<distributed_stream_loader_t>(_engine_loader.get_engine(), _engine_loader.get_id()),
        m_provider_id(_engine_loader.get_id()),
        task_type(_task_type), K(_K), N(_N), R(_R), C(_C), rand_gen(seed),
        num_samples_per_representative(_num_samples_per_representative),
        representative_shape(_representative_shape), buffer_strategy(_buffer_strategy), verbose(_verbose) {
    // 4 is the number of bytes in a float32
    num_bytes_per_representative = 4 * std::accumulate(representative_shape.begin(), representative_shape.end(), 1, std::multiplies<int>());

    m_server_procedure = define("get_samples", &distributed_stream_loader_t::get_remote_samples);
    // Register the remote procedure
    m_client_procedure = get_engine().define("get_samples");

    // Setup a finalization callback for this provider, in case it is
    // still alive when the engine is finalized.
    get_engine().push_finalize_callback(this, [p=this]() {
        if (p->verbose)
            DBG("[" << p->m_provider_id << "] Shutting down current provider...");
        delete p;
    });

    // MPI has maybe been initialized by horovodrun
    int mpi_initialized = true;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(NULL, NULL); 
        //throw std::runtime_error("MPI should be initialized outside of Neomem.");
    } else {
        mpi_was_initialized = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_num_workers);
    m_local_rank = std::atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

    // If enabled, get the remote endpoints via the MPI publishing mechanism
    if (discover_endpoints) {
        std::map<std::string, int> all_endpoints = gather_endpoints();
        if (all_endpoints.size() > 0) {
            std::cout << "endpoint size " << all_endpoints.size() << std::endl;
            register_endpoints(all_endpoints);
        }
    }

#ifndef WITHOUT_CUDA
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(m_local_rank % num_devices);

    if (verbose)
        DBG("[" << engine_loader.get_id() << "] Setting CUDA device " << m_local_rank % num_devices);

    CHECK_CUDA_ERROR(cudaStreamCreate(&m_streams[0])); // streamNonBlockingSync causes a sync issue
    // To reproduce, use only async copy functions, except in copy_last_batch.
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_streams[1]));
    CHECK_CUDA_ERROR(cudaStreamCreate(&m_streams[2]));
#endif

    init_rehearsal_buffers(torch::cuda::is_available());
    init_receiving_rdma_buffer();
}

/* static */ distributed_stream_loader_t* distributed_stream_loader_t::create(const engine_loader_t& engine_loader, Task task_type,
    unsigned int K, unsigned int N, unsigned int R, unsigned int C, int64_t seed,
    unsigned int num_samples_per_representative, std::vector<long> representative_shape,
    BufferStrategy buffer_strategy,
    bool discover_endpoints, bool verbose) {
    return new distributed_stream_loader_t(engine_loader, task_type, K, N, R, C, seed,
        num_samples_per_representative, representative_shape, buffer_strategy, discover_endpoints, verbose);
}

/**
 * Contact all other nodes to get their endpoints. The returned dictionary maps
 * endpoints (keys) to provider ids (values).
 */
std::map<std::string, int> distributed_stream_loader_t::gather_endpoints() {
    nvtx3::scoped_range nvtx{"gather_endpoints"};

    std::map<std::string, int> endpoints = {{get_engine().self(), m_provider_id}};
    auto all_endpoints = gather_dictionary(endpoints, m_num_workers);

    return all_endpoints;
}

/**
 *
 */
void distributed_stream_loader_t::register_endpoints(const std::map<std::string, int>& endpoints) {
    nvtx3::scoped_range nvtx{"register_endpoints"};

    for (auto endpoint : endpoints) {
        std::cout << "Looking up " << endpoint.first << ", " << endpoint.second << std::endl;
        tl::endpoint server = get_engine().lookup(endpoint.first);
        provider_handles.emplace_back(tl::provider_handle(server, endpoint.second));
    }
    ASSERT(static_cast<int>(provider_handles.size()) == m_num_workers);
}

/**
 * Initialize
 */
void distributed_stream_loader_t::init_rehearsal_buffers(bool pin_buffers) {
    nvtx3::scoped_range nvtx{"init_rehearsal_buffer"};

    auto size = K * N * num_samples_per_representative;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pin_buffers);

    auto rehearsal_shape = representative_shape;
    rehearsal_shape.insert(rehearsal_shape.begin(), size);
    rehearsal_tensor = new torch::Tensor(torch::empty(rehearsal_shape, options));
    ASSERT(rehearsal_tensor->is_contiguous());
    rehearsal_metadata.insert(rehearsal_metadata.begin(), K, std::make_pair(0, 1.0));
    rehearsal_counts.insert(rehearsal_counts.begin(), K, 0);

    if (verbose)
        DBG("[" << m_provider_id << "] Distributed buffer memory allocated!");

    // Initializing server bulks
    auto shape = representative_shape;
    shape.insert(shape.begin(), R);
    for (size_t r = 0; r < num_samples_per_representative; r++) {
        exposed_memory_t server_mem;
        server_mem.buffer = new torch::Tensor(torch::empty(shape, options));
        server_mem.segments.emplace_back(server_mem.buffer->data_ptr(), R * num_bytes_per_representative);
        server_mem.bulk = get_engine().expose(server_mem.segments, tl::bulk_mode::read_only);

        server_mems.emplace_back(std::move(server_mem));

        if (verbose)
            DBG("[" << m_provider_id << "] Server mem " << r << " initialized!");
    }

    if (verbose)
        DBG("[" << m_provider_id << "] Server mems initialized!");
}

/**
 * Should return a bulk, taking into account:
 * - the 'allocated policy'
 * - the buffer strategy, NoBuffer, CPUBuffer or CUDABuffer
 */
void distributed_stream_loader_t::init_receiving_rdma_buffer() {
    nvtx3::scoped_range nvtx{"init_receiving_rdma_buffer"};

    if (buffer_strategy == NoBuffer) {
        if (!m_use_allocated_variables)
            throw std::invalid_argument("NoBuffer policy is selected, so we should write in a variable declared on the Python side, which you didn't provide (or use CPUBuffer or CUDABuffer)");
        //if (!engine_loader.is_cuda_rdma_enabled() && alloc_aug_samples.is_cuda())
        //    throw std::invalid_argument("NoBuffer policy is selected, but cuda+verbs is not supported");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (buffer_strategy == CUDABuffer) {
        //if (!engine_loader.is_cuda_rdma_enabled())
        //    throw std::invalid_argument("CUDABuffer policy is selected, but cuda+verbs is not supported");

        options = options.device(torch::kCUDA);
    } else {
#ifndef WITHOUT_CUDA
        options = options.pinned_memory(true);
#endif
    }

    // Initializing client bulks    
    auto shape = representative_shape;
    shape.insert(shape.begin(), R);
    for (size_t r = 0; r < num_samples_per_representative; r++) {
        exposed_memory_t client_mem;
        client_mem.buffer = new torch::Tensor(torch::ones(shape, options));

        struct hg_bulk_attr attr;
        memset(&attr, 0, sizeof(attr));
        if (client_mem.buffer->is_cuda())
            attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_CUDA;
        else
            attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_HOST;

        client_mem.segments.emplace_back(client_mem.buffer->data_ptr(), R * num_bytes_per_representative);
        client_mem.bulk = get_engine().expose(client_mem.segments, tl::bulk_mode::write_only, attr);

        client_mems.emplace_back(std::move(client_mem));

        if (verbose)
            DBG("[" << m_provider_id << "] Client mem " << r << " initialized!");
    }

    if (verbose)
        DBG("[" << m_provider_id << "] Client mems initialized!");
}

/**
 * Start the thread disptaching rpcs to prepare the next augmented minibatch.
 */
void distributed_stream_loader_t::start() {
    // The thread executing the actual client issuing rpcs
    es = tl::xstream::create();
    async_thread = es->make_thread([this]() {
        async_process();
    });
    started = true;
}

/**
 * This method consumes the data pushed into the request_queue by accumulate(),
 * processes it, samples data from all other servers, and push the new data into
 * the response_queue (which will be consumed in turn by wait()).
 *
 * Should be executed in a dedicated thread.
 */
void distributed_stream_loader_t::async_process() {
    while (true) {
        nvtx3::mark("while start in async_process");

        // WAITING for data
        std::unique_lock<tl::mutex> lock(request_mutex);
        while (request_queue.empty())
            request_cond.wait(lock);
        auto batch = request_queue.front();
        request_queue.pop_front();
        lock.unlock();

        if (verbose)
            DBG("[" << m_provider_id << "] Consuming a batch!");

        nvtx3::mark("new iteration in async_process");

        // An empty batch is a signal for shutdown
        if (!batch.samples.defined())
            break;

        // Initialization of the augmented result, which changes at every
        // iteration with implementation `standard`
        if (!m_use_allocated_variables) {
            client_dest_tensors.clear();
            client_dest_tensors.push_back(&batch.aug_samples);
            dest_targets = &batch.aug_targets;
            dest_weights = &batch.aug_weights;
            for (size_t r = 0; r < num_samples_per_representative - 1; r++) {
                client_dest_tensors.push_back(&batch.aug_ground_truth[r]);
            }

            batch.m_augmentation_mark = batch.get_size();

            copy_last_batch(batch);
        }

        if (m_augmentation_enabled)
            augment_batch(batch);

        populate_rehearsal_buffer(batch);
        //update_representative_weights(R, batch_size);

#ifndef WITHOUT_CUDA
        {
            nvtx3::scoped_range nvtx{"backend_overhead"};
            // Wait for all async CUDA copies
            cudaStreamSynchronize(m_streams[0]);
        }
#endif

        /*
        // If using memcpy in copy_last_batch, this should be moved there
        for (size_t j = 0; j < batch.get_size(); j++) {
            ASSERT(torch::equal(batch.aug_samples[j][0][0][0], batch.aug_targets[j].to(torch::kFloat32)));
        }
        for (int j = 0; j < batch.aug_size; j++) {
            ASSERT(torch::equal(batch.aug_samples[j][0][0][0], batch.aug_targets[j].to(torch::kFloat32)));
        }
        */

        metadata.clear();
        i_batch++;

        nvtx3::mark("iteration has been processed in async_process");

        lock.lock();
        response_queue.emplace_back(batch);
        lock.unlock();
        request_cond.notify_one();
    }

    for (auto& provider_handle : provider_handles) {
        if (provider_handle.provider_id() == m_provider_id)
            get_engine().shutdown_remote_engine(provider_handle);
    }
}

/**
 * Copy incoming sample/target pairs and associated weights to the next
 * augmented minibatch.
 */
void distributed_stream_loader_t::copy_last_batch(const queue_item_t &batch) {
    nvtx3::scoped_range nvtx{"copy_last_batch"};
    if (verbose)
        DBG("[" << m_provider_id << "] Copying last batch!");

    // Copy incoming samples into the next augmented minibatch
#ifndef WITHOUT_CUDA
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        (char *) batch.aug_samples.data_ptr(),
        batch.samples.data_ptr(),
        batch.get_size() * num_bytes_per_representative,
        cudaMemcpyDefault,
        m_streams[0]
    ));
    for (size_t r = 0; r < num_samples_per_representative - 1; r++) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            (char *) batch.aug_ground_truth[r].data_ptr(),
            batch.ground_truth[r].data_ptr(),
            batch.get_size() * num_bytes_per_representative,
            cudaMemcpyDefault,
            m_streams[0]
        ));
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        (char *) batch.aug_targets.data_ptr(),
        batch.targets.data_ptr(),
        batch.get_size() * batch.targets.element_size(),
        cudaMemcpyDefault,
        m_streams[0]
    ));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        (char *) batch.aug_weights.data_ptr(),
        batch.weights.data_ptr(),
        batch.get_size() * batch.weights.element_size(),
        cudaMemcpyDefault,
        m_streams[0]
    ));
#else
    std::memcpy(
        (char *) batch.aug_samples.data_ptr(),
        batch.samples.data_ptr(),
        batch.get_size() * num_bytes_per_representative
    );
    for (size_t r = 0; r < num_samples_per_representative - 1; r++) {
        std::memcpy(
            (char *) batch.aug_ground_truth[r].data_ptr(),
            batch.ground_truth[r].data_ptr(),
            batch.get_size() * num_bytes_per_representative
        );
    }
    std::memcpy(
        (char *) batch.aug_targets.data_ptr(),
        batch.targets.data_ptr(),
        batch.get_size() * batch.targets.element_size()
    );
    std::memcpy(
        (char *) batch.aug_weights.data_ptr(),
        batch.weights.data_ptr(),
        batch.get_size() * batch.weights.element_size()
    );
#endif

    m_metrics[i_batch].batch_copy_time = 0;
}

/**
 * This function orchestrates the minibatch augmentation process, by
 * performing the following steps:
 *
 * 1) Dispatch rpcs to other processes. This is required by global sampling: a
 * subset of remote representatives are sampled to add diversity to the
 * augmented minibatch being constructed.
 *
 * 2) Wait for rpcs to resolve, and write weights and labels when consumed.
 * Depending if use_these_allocated_variables() has been called before, remote
 * data will be written in the `batch.aug_labels`/`batch.aug_weights` or
 * `alloc_aug_labels`/`alloc_aug_weights`. `dest_labels` and `dest_weights`
 * point to the correct variable.
 *
 * 3) Copy samples that have been written to the exposed memory to the
 * minibatch to augment `dest_samples`.
 */
void distributed_stream_loader_t::augment_batch(queue_item_t &batch) {
    nvtx3::scoped_range nvtx{"augment_batch"};

    // Dispatch rpcs to other processes
    std::vector<tl::async_response> responses;
    dispatch_rpcs(responses);
    resolve_rpcs(responses, batch);
}

/**
 * This function dispatches rpc requests to get R remote representatives.
 */
std::size_t distributed_stream_loader_t::dispatch_rpcs(std::vector<tl::async_response> &responses) {
    if (verbose)
        DBG("[" << m_provider_id << "] Dispatching rpcs");

    std::vector<tl::bulk> client_bulks;
    for (size_t i = 0; i < client_mems.size(); i++)
        client_bulks.push_back(client_mems[i].bulk);

    // Iterate over nodes and issuing corresponding rpc requests
    std::unordered_map<int, std::vector<int>> indices_per_node = pick_random_indices(R);
    auto j = 0;
    for (const auto& indices : indices_per_node) {
        tl::provider_handle& ph = provider_handles[indices.first];
        auto response = m_client_procedure.on(ph).async(client_bulks, indices.second, j);
        responses.push_back(std::move(response));

        // Valid because the offset and num_bytes_per_representatives are equal for each sample (including amp and ph)
        j += indices.second.size() * num_bytes_per_representative;
    }
    ASSERT(responses.size() == indices_per_node.size());

    m_metrics[i_batch].bulk_prepare_time = 0;
    return indices_per_node.size();
}

/**
 * Wait for rpc requests to resolve. The returned data is written in a buffer
 * representing the minibatch to augment, i.e., `batch.aug_samples` or
 * `alloc_aug_samples`.
 */
void distributed_stream_loader_t::resolve_rpcs(std::vector<tl::async_response>& responses, queue_item_t &batch) {
    if (verbose)
        DBG("[" << m_provider_id << "] Resolving rpcs...");

    // Sequence of integers representing sections that have been written,
    // first element is the memory offset, second is the number of bytes
    std::vector<std::pair<int, int>> memory_sections;

    for (size_t i = 0; i < responses.size(); i++) {
        // Waiting for rpcs to complete, careful to keep the order!
        std::vector<std::tuple<int, float, size_t, size_t>> m = responses[i].wait();
        // Store metadata so that it lives long enough for CUDA async transfers
        metadata.push_back(m);

        for (const auto &it : metadata.back()) {
            const int* label = &std::get<0>(it);
            const float* weight = &std::get<1>(it);
            const size_t* num_targets = &std::get<2>(it);
            const size_t* offset = &std::get<3>(it);

            if (*num_targets > 0) {
                memory_sections.emplace_back(*offset, *num_targets * num_bytes_per_representative);
            }

            for (size_t j = 0; j < *num_targets; j++) {
#ifndef WITHOUT_CUDA
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    // no * batch.element_size() as type is given
                    dest_targets->data_ptr<long int>() + batch.aug_size,
                    label,
                    sizeof(*label),
                    cudaMemcpyDefault,
                    m_streams[0]
                ));
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    dest_weights->data_ptr<float>() + batch.aug_size,
                    weight,
                    sizeof(*weight),
                    cudaMemcpyDefault,
                    m_streams[0]
                ));
#else
                dest_targets->index_put_({batch.aug_size}, *label);
                dest_weights->index_put_({batch.aug_size}, *weight);
#endif
                batch.aug_size++;
            }
        }
    }

    m_metrics[i_batch].rpcs_resolve_time = 0;

    // Maybe some nodes hadn't any data to return, so we have to prepare an
    // array of contiguous sections to transfer from receiving buffer to the
    // augmented minibatch.
    std::vector<std::pair<int, int>> contiguous_memory_sections = merge_contiguous_memory(memory_sections);

    // Copy representatives
    if (buffer_strategy != NoBuffer) {
        copy_exposed_buffer_to_aug_batch(batch, contiguous_memory_sections);
    }
}

/**
 * Returns a vector to keep the same order.
 */
std::vector<std::pair<int, int>> distributed_stream_loader_t::merge_contiguous_memory(std::vector<std::pair<int, int>>& sections) const {
    std::vector<std::pair<int, int>> mergedPairs;
    if (sections.empty())
        return mergedPairs;

    // Sort the sections based on the memory offset (first element of the pair)
    std::sort(sections.begin(), sections.end());

    // Merge contiguous chunks of memory
    mergedPairs.push_back(sections[0]);
    for (size_t i = 1; i < sections.size(); ++i) {
        int prevEnd = mergedPairs.back().first + mergedPairs.back().second;
        if (sections[i].first == prevEnd) {
            // Merge the current pair with the previous pair
            mergedPairs.back().second += sections[i].second;
        } else {
            // Non-contiguous, add as a new pair
            mergedPairs.push_back(sections[i]);
        }
    }

    return mergedPairs;
}

/**
 * This function copies data from the exposed bulk to the minibatch
 * `dest_samples`. The latter has either been passed during the last iteration
 * (`batch.aug_samples`)  or has been allocated once at the beginning of the
 * execution (`alloc_aug_samples`).
 */
void distributed_stream_loader_t::copy_exposed_buffer_to_aug_batch(const queue_item_t &batch, const std::vector<std::pair<int, int>>& sections) {
    nvtx3::scoped_range nvtx{"copy_exposed_buffer_to_aug_batch"};

    auto cumulated_offset = 0;
    for (const auto& pair : sections) {
        // First element is the offset in the client exposed memory, second
        // element is the chunk size in bytes.
#ifndef WITHOUT_CUDA
        for (size_t r = 0; r < num_samples_per_representative; r++) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                (char *) client_dest_tensors[r]->data_ptr() + batch.m_augmentation_mark * num_bytes_per_representative + cumulated_offset,
                (char *) client_mems[r].buffer->data_ptr() + pair.first,
                pair.second,
                cudaMemcpyDefault,
                m_streams[0]
            ));
        }
#else
        for (size_t r = 0; r < num_samples_per_representative; r++) {
            std::memcpy(
                (char *) client_dest_tensors[r]->data_ptr() + batch.m_augmentation_mark * num_bytes_per_representative + cumulated_offset,
                (char *) client_mems[r].buffer->data_ptr() + pair.first,
                pair.second
            );
        }
#endif
        cumulated_offset += pair.second;
    }

    m_metrics[i_batch].representatives_copy_time = 0;
}

/**
 * Selection without replacement from remote nodes + current node.
 *
 * The map returned by this function maps remote node indices to local indices.
 * Local indices might be used to access the provider_handles vector.
 */
std::unordered_map<int, std::vector<int>> distributed_stream_loader_t::pick_random_indices(int R) {
    nvtx3::scoped_range nvtx{"pick_random_indices"};

    const unsigned int max_global_index = provider_handles.size() * K * N;
    std::uniform_int_distribution<unsigned int> dice(0, max_global_index - 1);
    std::vector<unsigned int> choices(R);
    int i = 0;
    while (i < R) {
        int random_global_index = dice(rand_gen);
        if (std::find(choices.begin(), choices.end(), random_global_index) != choices.end())
            continue;
        choices[i++] = random_global_index;
    }

    // Map remote node indices to local indices
    std::unordered_map<int, std::vector<int>> indices_per_node;
    for (size_t i = 0; i < choices.size(); i++) {
        int global_index = choices[i];
        int local_index = global_index % (K * N);
        size_t node = global_index / (K * N);
        ASSERT(node >= 0 && node < provider_handles.size());
        indices_per_node[node].push_back(local_index);
    }

    return indices_per_node;
}

/*
 * Sample C random elements from the given batch to populate the rehearsal
 * buffer.
 */
void distributed_stream_loader_t::populate_rehearsal_buffer(const queue_item_t& batch) {
    nvtx3::scoped_range nvtx{"populate_rehearsal_buffer"};

    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    std::uniform_int_distribution<unsigned int> dice_candidate(0, batch.get_size() - 1);
    std::uniform_int_distribution<unsigned int> dice_buffer(0, N - 1);
    for (size_t i = 0; i < batch.get_size(); i++) {
        //if (dice(rand_gen) >= C)
        //    break;
        int label = 0;
        if (task_type == Classification)
            label = batch.targets[i].item<int>();

        size_t index = -1;
        if (rehearsal_metadata[label].first < N) {
            index = rehearsal_metadata[label].first;
        } else {
            if (dice_candidate(rand_gen) >= C)
                continue;
            index = dice_buffer(rand_gen);
        }

        size_t j = N * label + index;
        ASSERT(j < K * N);
#ifndef WITHOUT_CUDA
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            (char *) rehearsal_tensor->data_ptr() + num_samples_per_representative * num_bytes_per_representative * j,
            (char *) batch.samples.data_ptr() + num_bytes_per_representative * i,
            num_bytes_per_representative,
            cudaMemcpyDefault,
            m_streams[1]
        ));
        for (size_t r = 0; r < num_samples_per_representative - 1; r++) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                (char *) rehearsal_tensor->data_ptr() + num_samples_per_representative * num_bytes_per_representative * j + num_bytes_per_representative * (r + 1),
                (char *) batch.ground_truth[r].data_ptr() + num_bytes_per_representative * i,
                num_bytes_per_representative,
                cudaMemcpyDefault,
                m_streams[1]
            ));
        }
#else
        std::memcpy(
            (char *) rehearsal_tensor->data_ptr() + num_samples_per_representative * num_bytes_per_representative * j,
            (char *) batch.samples.data_ptr() + num_bytes_per_representative * i,
            num_bytes_per_representative
        );
        for (size_t r = 0; r < num_samples_per_representative - 1; r++) {
            std::memcpy(
                (char *) rehearsal_tensor->data_ptr() + num_samples_per_representative * num_bytes_per_representative * j + num_bytes_per_representative * (r + 1),
                (char *) batch.ground_truth[r].data_ptr() + num_bytes_per_representative * i,
                num_bytes_per_representative
            );
        }
#endif

        if (index >= rehearsal_metadata[label].first) {
            m_rehearsal_size++;
            rehearsal_metadata[label].first++;
        }
        rehearsal_counts[label]++;
    }

#ifndef WITHOUT_CUDA
    // The rehearsal_mutex is still held
    cudaStreamSynchronize(m_streams[1]);
#endif

    m_metrics[i_batch].buffer_update_time = 0;
}

/**
 * With big datasets like ImageNet, the following formula results in really
 * small weights. Keeping this function as future work.
 */
void distributed_stream_loader_t::update_representative_weights(const queue_item_t& batch, int num_representatives) {
    nvtx3::scoped_range nvtx{"update_representative_weights"};

    float weight = (float) batch.get_size() / (float) (num_representatives * m_rehearsal_size);
    for (size_t i = 0; i < rehearsal_metadata.size(); i++) {
        rehearsal_metadata[i].second = std::max(std::log(rehearsal_counts[i] * weight), 1.0f);
    }
}

/*
 * This function is invoked via a client rpc, thus executed on a server
 * instance.
 *
 * j is the offset where this server procedure should write into the client buffer.
 */
void distributed_stream_loader_t::get_remote_samples(const tl::request& req, std::vector<tl::bulk>& client_bulks, const std::vector<int>& indices, int client_bulks_offset) {
    nvtx3::scoped_range nvtx{"get_remote_samples"};

    int c = 0, o = 0;
    std::vector<std::tuple<size_t, float, std::vector<int>>> samples;

    /**
    * Input
    * Vector of indices
    *
    * Output
    * Rehearsal buffer, unordered map indexed by labels
    * - (label1, weight, reprs_indices)
    * - (label2, weight, reprs_indices)
    * If a representative is already present for a label, the representative
    * index is appended to repr_indices.
    **/
    if (m_rehearsal_size > 0) {
        for (auto index : indices) {
            size_t rehearsal_class_index = index / N;
            const int num_zeros = std::count_if(rehearsal_metadata.begin(), rehearsal_metadata.end(),
                [](const auto &p) { return p.first == 0; }
            );
            // We only consider classes with at least one element
            rehearsal_class_index %= (rehearsal_metadata.size() - num_zeros);

            size_t j = -1, i = 0;
            for (; i < rehearsal_metadata.size(); i++) {
                if (rehearsal_metadata[i].first == 0)
                    continue;
                j++;
                if (j == rehearsal_class_index)
                    break;
            }

            const size_t rehearsal_repr_of_class_index = (index % N) % rehearsal_metadata[i].first;

            if (std::none_of(samples.begin(), samples.end(), [&](const auto& el) { return std::get<0>(el) == i; })) {
                samples.emplace_back(i, rehearsal_metadata[i].second, std::vector<int>{});
            }
            for (auto& el : samples) {
                if (std::get<0>(el) == i) {
                    std::get<2>(el).push_back(i * N + rehearsal_repr_of_class_index);
                }
            }

            c++;
        }
    }

    if (verbose && c > 0) {
        std::cout << "[" << m_provider_id << "] Sending " << c << "/" << indices.size()  << " representatives from "
            << samples.size() << " different classes to remote node (endpoint: "
            << req.get_endpoint() << ", writing at offset " << client_bulks_offset << " for all " << client_bulks.size() << " client bulks)" << std::endl;
    }

    /**
    * Fill the RDMA buffer with tensors
    *
    * Input
    * Rehearsal buffer, unordered map indexed by labels
    * - (label1, weight, reprs_indices)
    * - (label2, weight, reprs_indices)
    *
    *
    * Output
    * Metadata, vector (to preserve the order) of tuples
    * - {(label1, weight, num_reprs), (label2, weight, num_reprs)}
    * Segments
    * - {(ptrrepA, nbytesA) (ptrrepB, nbytesB) (ptrrepC, nbytesC) (ptrrepD, nbytesD)}
    *
    * repA and repB are of label1, repC and repD are of label2
    **/
    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    std::vector<std::tuple<int, float, size_t, size_t>> attached_metadata;
    for (const auto &el : samples) {
        int label;
        float weight;
        std::vector<int> reprs_indices;
        std::tie(label, weight, reprs_indices) = el;

        attached_metadata.emplace_back(std::make_tuple(label, weight, reprs_indices.size(), client_bulks_offset + o * num_bytes_per_representative));

        for (size_t i = 0; i < reprs_indices.size(); i++) {
#ifndef WITHOUT_CUDA
            for (size_t r = 0; r < num_samples_per_representative; r++) {
                const int index = num_samples_per_representative * reprs_indices[i] + r;
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    (char *) server_mems[r].buffer->data_ptr() + num_bytes_per_representative * o,
                    rehearsal_tensor->index({index}).data_ptr(),
                    num_bytes_per_representative,
                    cudaMemcpyDefault,
                    m_streams[2]
                ));
            }
#else
            for (size_t r = 0; r < num_samples_per_representative; r++) {
                const int index = num_samples_per_representative * reprs_indices[i] + r;
                server_mems[r].buffer->index_put_({o}, rehearsal_tensor->index({index}));
            }
#endif
            o++;
        }
    }
    ASSERT(c == o);
    ASSERT(samples.size() == attached_metadata.size());

    if (c > 0) {
#ifndef WITHOUT_CUDA
        // The rehearsal_mutex is still held
        cudaStreamSynchronize(m_streams[2]);
#endif
        auto size = c * num_bytes_per_representative;
        for (size_t r = 0; r < num_samples_per_representative; r++) {
            server_mems[r].bulk(0, size) >> client_bulks[r](client_bulks_offset, size).on(req.get_endpoint());
        }
    }

    req.respond(attached_metadata);
}

/**
 * This is called from Python in a synchronous fashion. We push the incoming
 * data to the request_queue for it to be consumed by the client thread in an
 * asynchronous fashion.
 *
 * This version of accumulate corresponds to the `flyweight` buffer
 * implementation, i.e., use_these_allocated_variables() has been called
 * before and these given variables will be populated. The resulting
 * batch.aug_size will have a min value of 0.
 */
void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets) {
    nvtx3::scoped_range nvtx{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");
    if (!m_use_allocated_variables)
        throw std::runtime_error("You didn't pass variables to augment, so you should call use_these_allocated_variables() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets));
    lock.unlock();
    request_cond.notify_one();
}

/**
 * This is called from Python in a synchronous fashion. We push the incoming
 * data to the request_queue for it to be consumed by the client thread in an
 * asynchronous fashion.
 *
 * batch.aug_size will have a min value of the minibatch size defined in Python.
 */
void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    nvtx3::scoped_range nvtx{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets, aug_samples, aug_targets, aug_weights));
    lock.unlock();
    request_cond.notify_one();
}

void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets, std::vector<torch::Tensor> &ground_truth) {
    nvtx3::scoped_range nvtx{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");
    if (!m_use_allocated_variables)
        throw std::runtime_error("You didn't pass variables to augment, so you should call use_these_allocated_variables() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets, ground_truth));
    lock.unlock();
    request_cond.notify_one();
}

void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets, std::vector<torch::Tensor> &ground_truth,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights, std::vector<torch::Tensor> &aug_ground_truth) {
    nvtx3::scoped_range nvtx{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets, ground_truth, aug_samples, aug_targets, aug_weights, aug_ground_truth));
    lock.unlock();
    request_cond.notify_one();
}

/**
 * This function should be called when using the `flyweight` buffer implementation.
 */
void distributed_stream_loader_t::use_these_allocated_variables(const torch::Tensor &aug_samples,
                const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    alloc_aug_samples = aug_samples;
    alloc_aug_targets = aug_targets;
    alloc_aug_weights = aug_weights;

    client_dest_tensors.push_back(&alloc_aug_samples);
    dest_targets = &alloc_aug_targets;
    dest_weights = &alloc_aug_weights;

    m_use_allocated_variables = true;

    ASSERT(alloc_aug_samples.dim() > 0 && alloc_aug_targets.dim() == 1);
    R = alloc_aug_samples.sizes()[0];
    ASSERT(R > 0 && R == alloc_aug_targets.sizes()[0]
        && R == alloc_aug_weights.sizes()[0]);
}

/**
 * This function should be called when using the `flyweight` buffer implementation.
 */
 /*
void distributed_stream_loader_t::use_these_allocated_variables(const torch::Tensor &aug_samples, std::vector<torch::Tensor> &aug_ground_truth,
                const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    alloc_aug_samples = aug_samples;
    alloc_aug_targets = aug_targets;
    alloc_aug_weights = aug_weights;

    client_dest_tensors.push_back(&alloc_aug_samples);
    dest_targets = &alloc_aug_targets;
    dest_weights = &alloc_aug_weights;

    m_use_allocated_variables = true;

    ASSERT(alloc_aug_samples.dim() > 0 && alloc_aug_targets.dim() == 1);
    R = alloc_aug_samples.sizes()[0];
    ASSERT(R > 0 && R == alloc_aug_targets.sizes()[0]
        && R == alloc_aug_weights.sizes()[0]);
}
*/

/**
 * This is called from Python in a synchronous fashion. We consume the
 * data processed by the client thread. If no data is ready, we just wait,
 * blocking the Python thread.
 */
int distributed_stream_loader_t::wait() {
    nvtx3::scoped_range nvtx{"wait"};

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (response_queue.empty())
        request_cond.wait(lock);
    auto batch = response_queue.front();
    response_queue.pop_front();
    return batch.aug_size;
}

void distributed_stream_loader_t::enable_augmentation(bool state) {
    m_augmentation_enabled = state;
}

void distributed_stream_loader_t::measure_performance(bool state) {
    m_measure_performance = state;
}

size_t distributed_stream_loader_t::get_rehearsal_size() {
    return m_rehearsal_size;
}

std::vector<float> distributed_stream_loader_t::get_metrics(size_t i_batch) {
    if (!m_metrics.count(i_batch))
        return {};
    return m_metrics[i_batch].get_durations();
}

void distributed_stream_loader_t::finalize() {
    std::unique_lock<tl::mutex> lock(request_mutex);
    request_queue.push_back(queue_item_t());
    lock.unlock();
    request_cond.notify_one();

    if (verbose)
        DBG("[" << m_provider_id << "] Finalize signal sent...");
}

distributed_stream_loader_t::~distributed_stream_loader_t() noexcept {
    m_server_procedure.deregister();
    // Pop the finalize callback. If this destructor was called
    // from the finalization callback, there is nothing to pop
    get_engine().pop_finalize_callback(this);

    es->join();

#ifndef WITHOUT_CUDA
    for (int i = 0; i < 3; ++i) {
        cudaStreamSynchronize(m_streams[i]);
        cudaStreamDestroy(m_streams[i]);
    }
#endif
    delete rehearsal_tensor;
}

namespace cereal {
    template<typename A> void save(A& ar, const torch::Tensor& t) {
        std::stringstream ss;
        torch::save(t, ss);
        ar(ss.str());
    }

    template<typename A> void load(A& ar, torch::Tensor& t) {
        std::string s;
        ar(s);
        std::stringstream ss(s);
        torch::load(t, ss);
    }
}
