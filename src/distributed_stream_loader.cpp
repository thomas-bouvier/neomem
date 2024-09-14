#include "distributed_stream_loader.hpp"
#include "mpi_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <utility>

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
 * This factory method (as well as the private constructor) prevents users
 * from putting an instance of distributed_stream_loader_t on the stack.
 */
/* static */ distributed_stream_loader_t* distributed_stream_loader_t::create(const engine_loader_t& engine_loader, Task task_type,
        unsigned int K, unsigned int N, unsigned int C, int64_t seed,
        unsigned int R, unsigned int num_samples_per_representative, std::vector<long> representative_shape,
        unsigned int R_distillation, unsigned int num_samples_per_activation, std::vector<long> activation_shape,
        BufferStrategy buffer_strategy, bool discover_endpoints, bool half_precision, bool verbose)
{
    Config config {
        task_type,
        K,
        N,
        C,
        seed,
        R,
        num_samples_per_representative,
        std::move(representative_shape),
        R_distillation,
        num_samples_per_activation,
        std::move(activation_shape),
        buffer_strategy,
        discover_endpoints,
        half_precision,
        verbose
    };

    return new distributed_stream_loader_t(engine_loader, config);
}

/**
 * This constructor initializes the provider. This class is both a server and a
 * client. There are n clients and n servers. Each client can get data from the
 * n servers. (n x n relation).
 *
 * A client holds some data in a "rehearsal buffer". A client updates the
 * content of its rehearsal buffer by sampling from the n other rehearsal
 * buffers.
 */
distributed_stream_loader_t::distributed_stream_loader_t(
    const engine_loader_t& _engine_loader, 
    const Config& config)
    : tl::provider<distributed_stream_loader_t>(_engine_loader.get_engine(), _engine_loader.get_id())
    , m_provider_id(_engine_loader.get_id())
    , m_task_type(config.task_type)
    , K(config.K)
    , N(config.N)
    , C(config.C)
    , rand_gen(config.seed)
    , m_R(config.R)
    , m_num_samples_per_representative(config.num_samples_per_representative)
    , representative_shape(config.representative_shape)
    , m_R_distillation(config.R_distillation)
    , m_num_samples_per_activation(config.num_samples_per_activation)
    , m_activation_shape(config.activation_shape)
    , buffer_strategy(config.buffer_strategy)
    , m_half_precision(config.half_precision)
    , m_verbose(config.verbose)
{
    initialize_num_bytes_per_representative();
    register_procedures();

    // Setup a finalization callback for this provider
    get_engine().push_finalize_callback(this, [p=this]() {
        if (p->m_verbose) {
            DBG("[" << p->m_provider_id << "] Shutting down current provider...");
            std::cout << "shut" << std::endl;
        } 
        delete p;
    });

    initialize_mpi();
    // Discover endpoints if required
    if (config.discover_endpoints) {
        auto endpoints = gather_endpoints();
        register_endpoints(endpoints);
    }

    initialize_cuda();
    initialize_rehearsal_buffers();
    initialize_rdma_buffers();
}

/**
 * Calculates the number of bytes per representative based on the shape.
 */
void distributed_stream_loader_t::initialize_num_bytes_per_representative() {
    m_num_bytes_per_representative = 4 * std::accumulate(representative_shape.begin(), representative_shape.end(), 1, std::multiplies<int>());
}

/**
 * Registers the RPC procedures for getting representatives and activations.
 */
void distributed_stream_loader_t::register_procedures() {
    m_server_representatives_procedure = define("get_representatives", &distributed_stream_loader_t::get_remote_representatives);
    m_server_activations_procedure = define("get_activations", &distributed_stream_loader_t::get_remote_activations);

    m_client_representatives_procedure = get_engine().define("get_representatives");
    m_client_activations_procedure = get_engine().define("get_activations");
}

/**
 * Initializes MPI, sets the rank and number of workers.
 */
void distributed_stream_loader_t::initialize_mpi() {
    int mpi_initialized = true;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        throw std::runtime_error("MPI should be initialized outside of Neomem.");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_num_workers);
}

/**
 * Initializes CUDA devices and streams if CUDA is available.
 */
void distributed_stream_loader_t::initialize_cuda() {
#ifndef WITHOUT_CUDA
    m_local_rank = std::atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(m_local_rank % num_devices);

    if (m_verbose)
        DBG("[" << m_provider_id << "] Setting CUDA device " << m_local_rank % num_devices);

    // streamNonBlockingSync causes a sync issue
    // To reproduce, use only async copy functions, except in copy_last_batch.
    for (size_t i = 0; i < m_streams.size(); i++)
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_streams[i]));
#endif
}

/**
 * Initializes the rehearsal buffers for representatives.
 */
void distributed_stream_loader_t::initialize_rehearsal_buffers() {
    init_rehearsal_buffers(
        m_rehearsal_representatives,
        m_num_samples_per_representative,
        representative_shape,
        torch::cuda::is_available()
    );
}

/**
 * Initializes the RDMA buffers for representatives and activations.
 */
void distributed_stream_loader_t::initialize_rdma_buffers() {
    if (m_task_type == Task::REHEARSAL || m_task_type == Task::REHEARSAL_KD) {
        init_receiving_rdma_buffer(
            m_server_mems,
            m_client_mems,
            m_R,
            m_num_samples_per_representative,
            representative_shape
        );
    }

    if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
        ASSERT(m_num_samples_per_activation > 0);
        // This should be changed to float32 if not training in half precision!
        m_num_bytes_per_activation = std::accumulate(m_activation_shape.begin(), m_activation_shape.end(), 1, std::multiplies<int>());
        if (m_half_precision) {
            m_num_bytes_per_activation *= 2;
        } else {
            m_num_bytes_per_activation *= 4;
        }

        init_rehearsal_buffers(
            m_rehearsal_activations,
            m_num_samples_per_activation,
            m_activation_shape,
            torch::cuda::is_available()
        );

        // Initializing bulks
        init_receiving_rdma_buffer(
            m_server_activations_mem,
            m_client_activations_mem,
            m_R_distillation,
            m_num_samples_per_activation,
            m_activation_shape
        );
        init_receiving_rdma_buffer(
            m_server_activations_rep_mem,
            m_client_activations_rep_mem,
            m_R_distillation,
            1,
            representative_shape
        );
    }
}

/**
 * Contact all other nodes to get their endpoints. The returned dictionary maps
 * endpoints (keys) to provider ids (values).
 */
std::map<std::string, int> distributed_stream_loader_t::gather_endpoints()
{
    nvtx3::scoped_range nvtx{"gather_endpoints"};

    std::map<std::string, int> endpoints = {{get_engine().self(), m_provider_id}};
    auto all_endpoints = gather_dictionary(endpoints, m_num_workers);

    return all_endpoints;
}

/**
 * Populate the provider_handles vector to access remote nodes.
 *
 * First handle corresponds to the local (current) server.
 */
void distributed_stream_loader_t::register_endpoints(const std::map<std::string, int>& endpoints)
{
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
void distributed_stream_loader_t::init_rehearsal_buffers(
        std::unique_ptr<torch::Tensor>& storage, size_t nsamples, std::vector<long> sample_shape, bool pin_buffers)
{
    nvtx3::scoped_range nvtx{"init_rehearsal_buffer"};

    auto size = K * N * nsamples;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pin_buffers);

    sample_shape.insert(sample_shape.begin(), size);
    storage = std::make_unique<torch::Tensor>(torch::empty(sample_shape, options));
    ASSERT(storage->is_contiguous());
    rehearsal_metadata.insert(rehearsal_metadata.begin(), K, std::make_pair(0, 1.0));
    rehearsal_counts.insert(rehearsal_counts.begin(), K, 0);

    if (m_verbose)
        DBG("[" << m_provider_id << "] Distributed buffer memory allocated!");
}

/**
 * Should return a bulk, taking into account:
 * - the 'allocated policy'
 * - the buffer strategy, NoBuffer, CPUBuffer or CUDABuffer
 */
void distributed_stream_loader_t::init_receiving_rdma_buffer(
    std::vector<exposed_memory_t>& server_mems, std::vector<exposed_memory_t>& client_mems, size_t nelements, size_t nsamples_per_element, std::vector<long> sample_shape)
{
    nvtx3::scoped_range nvtx{"init_receiving_rdma_buffer"};

    if (buffer_strategy == NoBuffer) {
        if (!m_use_allocated_variables) {
            throw std::invalid_argument("NoBuffer policy is selected, so we should write in a variable declared on the Python side, which you didn't provide (or use CPUBuffer or CUDABuffer)");
        }
        //if (!engine_loader.is_cuda_rdma_enabled() && alloc_aug_samples.is_cuda()) {
        // throw std::invalid_argument("NoBuffer policy is selected, but cuda+verbs is not supported");
        //}
    }

    //if (buffer_strategy == CUDABuffer && !engine_loader.is_cuda_rdma_enabled()) {
        // throw std::invalid_argument("CUDABuffer policy is selected, but cuda+verbs is not supported");
    //}

    // Initializing server bulks
    struct exposed_memory_attr server_attr;
    memset(&server_attr, 0, sizeof(server_attr));
    server_attr.bulk_mode = tl::bulk_mode::read_only;
    create_exposed_memory(server_mems, nelements, nsamples_per_element, sample_shape, server_attr);

    if (m_verbose)
        DBG("[" << m_provider_id << "] Server mems initialized!");

    // Initializing client bulk
    struct exposed_memory_attr client_attr;
    memset(&client_attr, 0, sizeof(client_attr));
    client_attr.cuda = buffer_strategy == CUDABuffer;
    client_attr.bulk_mode = tl::bulk_mode::write_only;
    create_exposed_memory(client_mems, nelements, nsamples_per_element, sample_shape, client_attr);

    if (m_verbose)
        DBG("[" << m_provider_id << "] Client mems initialized!");
}

/**
 * Create `exposed_memory_t` objects.
 */
void distributed_stream_loader_t::create_exposed_memory(
        std::vector<exposed_memory_t>& exposed_memory, size_t nelements, size_t nsamples_per_element, std::vector<long> sample_shape, exposed_memory_attr attr)
{
    auto nbytes = 4 * std::accumulate(sample_shape.begin(), sample_shape.end(), 1, std::multiplies<int>());
    sample_shape.insert(sample_shape.begin(), nelements);

    for (size_t r = 0; r < nsamples_per_element; r++) {
        exposed_memory_t memory;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        if (attr.cuda) {
            options = options.device(torch::kCUDA);
        } else {
#ifndef WITHOUT_CUDA
            options = options.pinned_memory(true);
#endif
        }
        memory.buffer = std::make_unique<torch::Tensor>(torch::ones(sample_shape, options));

        // Preparing Mercury attributes.
        struct hg_bulk_attr hg_attr;
        memset(&hg_attr, 0, sizeof(hg_attr));
        if (attr.cuda)
            hg_attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_CUDA;
        else
            hg_attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_HOST;

        memory.segments.emplace_back(memory.buffer->data_ptr(), nelements * nbytes);
        memory.bulk = get_engine().expose(memory.segments, attr.bulk_mode, hg_attr);

        exposed_memory.emplace_back(std::move(memory));
    }

    if (m_verbose)
        DBG("[" << m_provider_id << "] Initialized " << nsamples << " exposed memory regions!");
}

/**
 * Start the thread disptaching rpcs to prepare the next augmented minibatch.
 */
void distributed_stream_loader_t::start()
{
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
void distributed_stream_loader_t::async_process()
{
    while (true) {
        nvtx3::mark("while start in async_process");

        // WAITING for data
        std::unique_lock<tl::mutex> lock(request_mutex);
        while (request_queue.empty())
            request_cond.wait(lock);
        auto batch = request_queue.front();
        request_queue.pop_front();
        lock.unlock();

        if (m_verbose)
            DBG("[" << m_provider_id << "] Consuming a batch!");

        nvtx3::mark("new iteration in async_process");

        // An empty batch is a signal for shutdown
        if (batch.m_representatives.size() == 0) {
            break;
        }

        // Initialization of the augmented result, which changes at every
        // iteration with implementation `standard`
        if (!m_use_allocated_variables) {
            if (m_task_type == Task::REHEARSAL || m_task_type == Task::REHEARSAL_KD) {
                m_buf_representatives = std::make_shared<std::vector<torch::Tensor>>(batch.m_aug_representatives);
                m_buf_targets = std::make_shared<torch::Tensor>(batch.m_aug_targets);
                m_buf_weights = std::make_shared<torch::Tensor>(batch.m_aug_weights);

                ASSERT(m_buf_representatives->size() == m_num_samples_per_representative);

                batch.m_augmentation_mark = batch.get_size();
                copy_last_batch(batch);
            }

            if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) { 
                m_buf_activations = std::make_shared<std::vector<torch::Tensor>>(batch.m_buf_activations);
                m_buf_activations_rep = std::make_shared<torch::Tensor>(batch.m_buf_activations_rep);

                ASSERT(m_buf_activations->size() == m_num_samples_per_activation);
            }
        }

        if (m_augmentation_enabled) {
            augment_batch(batch);
        }

        populate_rehearsal_buffer(batch);
        //update_representative_weights(R, batch_size);

#ifndef WITHOUT_CUDA
        {
            nvtx3::scoped_range nvtx{"backend_overhead"};
            // Wait for all async CUDA copies
            cudaStreamSynchronize(m_streams[0]);
        }
#endif

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
void distributed_stream_loader_t::copy_last_batch(const queue_item_t &batch)
{
    nvtx3::scoped_range nvtx{"copy_last_batch"};
    if (m_verbose)
        DBG("[" << m_provider_id << "] Copying last batch!");

    // Copy incoming samples into the next augmented minibatch
#ifndef WITHOUT_CUDA
    for (size_t i = 0; i < m_num_samples_per_representative; i++) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            (char *) batch.m_aug_representatives[i].data_ptr(),
            batch.m_representatives[i].data_ptr(),
            batch.get_size() * m_num_bytes_per_representative,
            cudaMemcpyDefault,
            m_streams[0]
        ));
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        (char *) batch.m_aug_targets.data_ptr(),
        batch.m_targets.data_ptr(),
        batch.get_size() * batch.m_targets.element_size(),
        cudaMemcpyDefault,
        m_streams[0]
    ));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        (char *) batch.m_aug_weights.data_ptr(),
        batch.m_weights.data_ptr(),
        batch.get_size() * batch.m_weights.element_size(),
        cudaMemcpyDefault,
        m_streams[0]
    ));
#else
    for (size_t i = 0; i < m_num_samples_per_representative; i++) {
        std::memcpy(
            (char *) batch.m_aug_representatives[i].data_ptr(),
            batch.m_representatives[i].data_ptr(),
            batch.get_size() * m_num_bytes_per_representative
        );
    }
    std::memcpy(
        (char *) batch.m_aug_targets.data_ptr(),
        batch.m_targets.data_ptr(),
        batch.get_size() * batch.m_targets.element_size()
    );
    std::memcpy(
        (char *) batch.m_aug_weights.data_ptr(),
        batch.m_weights.data_ptr(),
        batch.get_size() * batch.m_weights.element_size()
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
void distributed_stream_loader_t::augment_batch(queue_item_t &batch)
{
    nvtx3::scoped_range nvtx{"augment_batch"};

    // Dispatch rpcs to other processes
    std::vector<tl::async_response> responses;
    dispatch_rpcs(responses);
    resolve_rpcs(responses, batch);
}

/**
 * This function dispatches rpc requests to get R remote representatives.
 */
void distributed_stream_loader_t::dispatch_rpcs(std::vector<tl::async_response> &responses)
{
    if (m_verbose)
        DBG("[" << m_provider_id << "] Dispatching rpcs");

    size_t nindices = 0;
    if (m_task_type == Task::REHEARSAL || m_task_type == Task::REHEARSAL_KD) {
        std::vector<tl::bulk> client_bulks;
        for (size_t i = 0; i < m_client_mems.size(); i++) {
            client_bulks.push_back(m_client_mems[i].bulk);
        }

        // Iterate over nodes and issuing corresponding rpc requests
        std::unordered_map<int, std::vector<int>> representatives_indices_per_node = pick_random_indices(m_R);

        auto offset = 0;
        for (const auto& indices : representatives_indices_per_node) {
            tl::provider_handle& ph = provider_handles[indices.first];

            auto response = m_client_representatives_procedure.on(ph).async(
                client_bulks, indices.second, offset
            );
            responses.push_back(std::move(response));

            // Valid because the offset and num_bytes_per_representative are equal for each sample (including amp and ph)
            offset += indices.second.size();
        }
        nindices += representatives_indices_per_node.size();
    }

    if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
        std::vector<tl::bulk> client_activations_bulks, client_activations_rep_bulks;
        for (size_t i = 0; i < m_client_activations_mem.size(); i++) {
            client_activations_bulks.push_back(m_client_activations_mem[i].bulk);
        }
        for (size_t i = 0; i < m_client_activations_rep_mem.size(); i++) {
            client_activations_rep_bulks.push_back(m_client_activations_rep_mem[i].bulk);
        }

        std::unordered_map<int, std::vector<int>> activations_indices_per_node = pick_random_indices(m_R_distillation);

        auto offset = 0;
        for (const auto& indices : activations_indices_per_node) {
            tl::provider_handle& ph = provider_handles[indices.first];

            auto response = m_client_activations_procedure.on(ph).async(
                client_activations_bulks, client_activations_rep_bulks, indices.second, offset
            );
            responses.push_back(std::move(response));

            // Valid because the offset and num_bytes_per_representative are equal for each sample (including amp and ph)
            offset += indices.second.size();
        }
        nindices += activations_indices_per_node.size();
    }

    ASSERT(responses.size() == nindices);

    m_metrics[i_batch].bulk_prepare_time = 0;
}

/**
 * Wait for rpc requests to resolve. The returned data is written in a buffer
 * representing the minibatch to augment, i.e., `batch.aug_samples` or
 * `alloc_aug_samples`.
 */
void distributed_stream_loader_t::resolve_rpcs(std::vector<tl::async_response>& responses, queue_item_t &batch)
{
    if (m_verbose)
        DBG("[" << m_provider_id << "] Resolving rpcs...");

    // Sequence of integers representing sections that have been written,
    // first element is the memory offset, second is the number of bytes
    std::vector<std::pair<int, int>> memory_sections;

    for (auto& response : responses) {
        // Waiting for rpcs to complete, careful to keep the order!
        // Store metadata so that it lives long enough for CUDA async transfers
        metadata.emplace_back(response.wait());

        for (const rpc_response_t& it : metadata.back()) {
            if (it.m_type == RPCResponseType::Representative) {
                if (it.m_num_elements > 0) {
                    memory_sections.emplace_back(it.m_offset, it.m_num_elements);
                }

                for (size_t j = 0; j < it.m_num_elements; j++) {
#ifndef WITHOUT_CUDA
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(
                        // no * batch.element_size() as type is given
                        m_buf_targets->data_ptr<long int>() + batch.aug_size,
                        &it.m_label,
                        sizeof(it.m_label),
                        cudaMemcpyDefault,
                        m_streams[0]
                    ));
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(
                        m_buf_weights->data_ptr<float>() + batch.aug_size,
                        &it.m_weight,
                        sizeof(it.m_weight),
                        cudaMemcpyDefault,
                        m_streams[0]
                    ));
#else
                    //m_buf_targets->index_put_({batch.aug_size}, it.m_label);
                    //m_buf_weights->index_put_({batch.aug_size}, it.m_weight);
                    std::memcpy(
                        // no * batch.element_size() as type is given
                        m_buf_targets->data_ptr<long int>() + batch.aug_size,
                        &it.m_label,
                        sizeof(it.m_label)
                    );
                    std::memcpy(
                        m_buf_weights->data_ptr<float>() + batch.aug_size,
                        &it.m_weight,
                        sizeof(it.m_weight)
                    );
#endif
                    batch.aug_size++;
                }
            } else if (it.m_type == RPCResponseType::Activation) {
                for (size_t j = 0; j < it.m_num_elements; j++) {
                    batch.activations_size++;
                }
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
        copy_exposed_buffer_to_python_batch(batch, contiguous_memory_sections);
    }
}

/**
 * Returns a vector to keep the same order.
 */
std::vector<std::pair<int, int>> distributed_stream_loader_t::merge_contiguous_memory(std::vector<std::pair<int, int>>& sections) const
{
    std::vector<std::pair<int, int>> mergedPairs;
    if (sections.empty()) {
        return mergedPairs;
    }

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
void distributed_stream_loader_t::copy_exposed_buffer_to_python_batch(const queue_item_t &batch, const std::vector<std::pair<int, int>>& sections)
{
    nvtx3::scoped_range nvtx{"copy_exposed_buffer_to_python_batch"};

    auto cumulated_offset = 0;
    for (const auto& pair : sections) {
        // First element is the offset in the client exposed memory, second
        // element is the chunk size in bytes.
#ifndef WITHOUT_CUDA
        if (m_task_type == Task::REHEARSAL || m_task_type == Task::REHEARSAL_KD) {
            for (size_t r = 0; r < m_num_samples_per_representative; r++) {
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    static_cast<char *>(m_buf_representatives->at(r).data_ptr()) + (batch.m_augmentation_mark + cumulated_offset) * m_num_bytes_per_representative,
                    static_cast<char *>(m_client_mems[r].buffer->data_ptr()) + pair.first * m_num_bytes_per_representative,
                    pair.second * m_num_bytes_per_representative,
                    cudaMemcpyDefault,
                    m_streams[0]
                ));
            }
        }

        if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
            // Copying activations
            for (size_t r = 0; r < m_num_samples_per_activation; r++) {
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    static_cast<char *>(m_buf_activations->at(r).data_ptr()) + cumulated_offset * m_num_bytes_per_activation,
                    static_cast<char *>(m_client_activations_mem[r].buffer->data_ptr()) + pair.first * m_num_bytes_per_activation,
                    pair.second * m_num_bytes_per_activation,
                    cudaMemcpyDefault,
                    m_streams[0]
                ));
            }
            // Copying corresponding training representative
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                static_cast<char *>(m_buf_activations_rep->data_ptr()) + cumulated_offset * m_num_bytes_per_representative,
                static_cast<char *>(m_client_activations_rep_mem[0].buffer->data_ptr()) + pair.first * m_num_bytes_per_representative,
                pair.second * m_num_bytes_per_representative,
                cudaMemcpyDefault,
                m_streams[0]
            ));
        }
#else
        if (m_task_type == Task::REHEARSAL || m_task_type == Task::REHEARSAL_KD) {
            for (size_t r = 0; r < m_num_samples_per_representative; r++) {
                std::memcpy(
                    static_cast<char *>(m_buf_representatives->at(r).data_ptr()) + (batch.m_augmentation_mark + cumulated_offset) * m_num_bytes_per_representative,
                    static_cast<char *>(m_client_mems[r].buffer->data_ptr()) + pair.first * m_num_bytes_per_representative,
                    pair.second * m_num_bytes_per_representative
                );
            }
        }

        if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
            // Copying activations
            for (size_t r = 0; r < m_num_samples_per_activation; r++) {
                std::memcpy(
                    static_cast<char *>(m_buf_activations->at(r).data_ptr()) + cumulated_offset * m_num_bytes_per_activation,
                    static_cast<char *>(m_client_activations_mem[r].buffer->data_ptr()) + pair.first * m_num_bytes_per_activation,
                    pair.second * m_num_bytes_per_activation
                );
            }
            // Copying corresponding training representative
            std::memcpy(
                static_cast<char *>(m_buf_activations_rep->data_ptr()) + cumulated_offset * m_num_bytes_per_representative,
                static_cast<char *>(m_client_activations_rep_mem[0].buffer->data_ptr()) + pair.first * m_num_bytes_per_representative,
                pair.second * m_num_bytes_per_representative
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
std::unordered_map<int, std::vector<int>> distributed_stream_loader_t::pick_random_indices(int R)
{
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
void distributed_stream_loader_t::populate_rehearsal_buffer(const queue_item_t& batch)
{
    nvtx3::scoped_range nvtx{"populate_rehearsal_buffer"};

    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    std::uniform_int_distribution<unsigned int> dice_candidate(0, batch.get_size() - 1);
    std::uniform_int_distribution<unsigned int> dice_buffer(0, N - 1);
    for (size_t i = 0; i < batch.get_size(); i++) {
        //if (dice(rand_gen) >= C)
        //    break;
        int label = (K == 1) ? 0 : batch.m_targets[i].item<int>();

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
        for (size_t k = 0; k < m_num_samples_per_representative; k++) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                (char *) m_rehearsal_representatives->data_ptr() + m_num_bytes_per_representative * (m_num_samples_per_representative * j + k),
                (char *) batch.m_representatives[k].data_ptr() + m_num_bytes_per_representative * i,
                m_num_bytes_per_representative,
                cudaMemcpyDefault,
                m_streams[1]
            ));
        }
        if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
            for (size_t k = 0; k < m_num_samples_per_activation; k++) {
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    (char *) m_rehearsal_activations->data_ptr() + m_num_bytes_per_activation * (m_num_samples_per_activation * j + k),
                    (char *) batch.m_activations[k].data_ptr() + m_num_bytes_per_activation * i,
                    m_num_bytes_per_activation,
                    cudaMemcpyDefault,
                    m_streams[1]
                ));
            }
        }
#else
        for (size_t k = 0; k < m_num_samples_per_representative; k++) {
            std::memcpy(
                (char *) m_rehearsal_representatives->data_ptr() + m_num_bytes_per_representative * (m_num_samples_per_representative * j + k),
                (char *) batch.m_representatives[k].data_ptr() + m_num_bytes_per_representative * i,
                m_num_bytes_per_representative
            );
        }
        if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
            for (size_t k = 0; k < m_num_samples_per_activation; k++) {
                std::memcpy(
                    (char *) m_rehearsal_activations->data_ptr() + m_num_bytes_per_activation * (m_num_samples_per_activation * j + k),
                    (char *) batch.m_activations[k].data_ptr() + m_num_bytes_per_activation * i,
                    m_num_bytes_per_activation
                );
            }
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
void distributed_stream_loader_t::update_representative_weights(const queue_item_t& batch, int num_representatives)
{
    nvtx3::scoped_range nvtx{"update_representative_weights"};

    float weight = (float) batch.get_size() / (float) (num_representatives * m_rehearsal_size);
    for (size_t i = 0; i < rehearsal_metadata.size(); i++) {
        rehearsal_metadata[i].second = std::max(std::log(rehearsal_counts[i] * weight), 1.0f);
    }
}

/**
 * Input
 * Vector of indices
 *
 * Output
 * Rehearsal buffer, unordered map indexed by labels
 * - (label1, weight, reprs_indices)
 * - (label2, weight, reprs_indices)
 *
 * If a representative is already present for a label, the representative
 * index is appended to repr_indices.
 */
std::vector<std::tuple<size_t, float, std::vector<int>>> distributed_stream_loader_t::get_actual_rehearsal_indices(const std::vector<int>& indices) const {
    std::vector<std::tuple<size_t, float, std::vector<int>>> samples;

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
        }
    }

    return samples;
}

int distributed_stream_loader_t::count_samples(const std::vector<std::tuple<size_t, float, std::vector<int>>>& samples) const {
    size_t nsamples = 0;

    std::for_each(samples.begin(), samples.end(), [&nsamples](const auto& sample) {
        const auto& indices = std::get<2>(sample);
        nsamples += indices.size();
    });

    return nsamples;
}

/*
 * This function is invoked via a client rpc, thus executed on a server
 * instance. Fill the RDMA buffer with representative tensors.
 *
 * j is the offset where this server procedure should write into the client buffer.
 *
 * Input
 * Rehearsal buffer, unordered map indexed by labels
 * - (label1, weight, reprs_indices)
 * - (label2, weight, reprs_indices)
 *
 * Output
 * Metadata, vector (to preserve the order) of tuples
 * - {(label1, weight, num_reprs), (label2, weight, num_reprs)}
 * Segments
 * - {(ptrrepA, nbytesA) (ptrrepB, nbytesB) (ptrrepC, nbytesC) (ptrrepD, nbytesD)}
 *
 * repA and repB are of label1, repC and repD are of label2
 */
void distributed_stream_loader_t::get_remote_representatives(
        const tl::request& req,
        std::vector<tl::bulk>& client_bulks, const std::vector<int>& indices, int offset)
{
    nvtx3::scoped_range nvtx{"get_remote_representatives"};

    std::vector<std::tuple<size_t, float, std::vector<int>>> samples = get_actual_rehearsal_indices(indices);
    const size_t nrepresentatives = count_samples(samples);

    if (m_verbose && nrepresentatives > 0) {
        std::cout << "[" << m_provider_id << "] Sending " << nrepresentatives << "/" << indices.size()  << " representatives from "
            << samples.size() << " different classes to remote node (endpoint: "
            << req.get_endpoint() << ", writing at offset " << offset << " for all " << client_bulks.size() << " client bulks)" << std::endl;
    }

    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    size_t o = 0;
    std::vector<rpc_response_t> attached_metadata;
    for (const auto &el : samples) {
        int label;
        float weight;
        std::vector<int> reprs_indices;
        std::tie(label, weight, reprs_indices) = el;

        attached_metadata.emplace_back(rpc_response_t(RPCResponseType::Representative, reprs_indices.size(), offset + o, label, weight));

        for (size_t i = 0; i < reprs_indices.size(); i++) {
#ifndef WITHOUT_CUDA
            for (size_t r = 0; r < m_num_samples_per_representative; r++) {
                const int index = m_num_samples_per_representative * reprs_indices[i] + r;
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    (char *) m_server_mems[r].buffer->data_ptr() + m_num_bytes_per_representative * o,
                    m_rehearsal_representatives->index({index}).data_ptr(),
                    m_num_bytes_per_representative,
                    cudaMemcpyDefault,
                    m_streams[2]
                ));
            }
#else
            for (size_t r = 0; r < m_num_samples_per_representative; r++) {
                const int index = m_num_samples_per_representative * reprs_indices[i] + r;
                std::memcpy(
                    (char *) m_server_mems[r].buffer->data_ptr() + m_num_bytes_per_representative * o,
                    m_rehearsal_representatives->index({index}).data_ptr(),
                    m_num_bytes_per_representative
                );
            }
#endif
            o++;
        }
    }
    ASSERT(nrepresentatives == o);
    ASSERT(samples.size() == attached_metadata.size());

    if (nrepresentatives > 0) {
#ifndef WITHOUT_CUDA
        // The rehearsal_mutex is still held
        cudaStreamSynchronize(m_streams[2]);
#endif

        const auto num_bytes_representatives = nrepresentatives * m_num_bytes_per_representative;

        for (size_t r = 0; r < m_num_samples_per_representative; r++) {
            m_server_mems[r].bulk(0, num_bytes_representatives)
                >> client_bulks[r](offset * m_num_bytes_per_representative, num_bytes_representatives).on(req.get_endpoint());
        }
    }

    req.respond(attached_metadata);
}

/*
 * This function is invoked via a client rpc, thus executed on a server
 * instance. Fill the RDMA buffer with activation tensors.
 *
 * j is the offset where this server procedure should write into the client buffer.
 *
 * Input
 * Rehearsal buffer, unordered map indexed by labels
 * - (label1, weight, reprs_indices)
 * - (label2, weight, reprs_indices)
 *
 * Output
 * Metadata, vector (to preserve the order) of tuples
 * - {(label1, weight, num_reprs), (label2, weight, num_reprs)}
 * Segments
 * - {(ptrrepA, nbytesA) (ptrrepB, nbytesB) (ptrrepC, nbytesC) (ptrrepD, nbytesD)}
 *
 * repA and repB are of label1, repC and repD are of label2
 */
void distributed_stream_loader_t::get_remote_activations(
        const tl::request& req,
        std::vector<tl::bulk>& client_activations_bulks, std::vector<tl::bulk>& client_activations_rep_bulks, const std::vector<int>& indices, int offset)
{
    nvtx3::scoped_range nvtx{"get_remote_activations"};

    std::vector<std::tuple<size_t, float, std::vector<int>>> samples = get_actual_rehearsal_indices(indices);
    const size_t nactivations = count_samples(samples);

    if (m_verbose && nactivations > 0) {
        std::cout << "[" << m_provider_id << "] Sending " << nactivations << "/" << indices.size()  << " activations and associated representatives from "
            << samples.size() << " different classes to remote node (endpoint: "
            << req.get_endpoint() << ", writing at offset " << offset << " for all " << client_activations_bulks.size() << " client activation bulks)" << std::endl;
    }

    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    size_t o = 0;
    std::vector<rpc_response_t> attached_metadata;
    for (const auto &el : samples) {
        std::vector<int> reprs_indices;
        std::tie(std::ignore, std::ignore, reprs_indices) = el;

        attached_metadata.emplace_back(rpc_response_t(RPCResponseType::Activation, reprs_indices.size(), offset + o));

        for (size_t i = 0; i < reprs_indices.size(); i++) {
#ifndef WITHOUT_CUDA
            for (size_t r = 0; r < m_num_samples_per_activation; r++) {
                const int index = m_num_samples_per_activation * reprs_indices[i] + r;
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    (char *) m_server_activations_mem[r].buffer->data_ptr() + m_num_bytes_per_activation * o,
                    (char *) m_rehearsal_activations->index({index}).data_ptr(),
                    m_num_bytes_per_activation,
                    cudaMemcpyDefault,
                    m_streams[3]
                ));
            }
            const int index = m_num_samples_per_representative * reprs_indices[i];
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                (char *) m_server_activations_rep_mem[0].buffer->data_ptr() + m_num_bytes_per_representative * o,
                (char *) m_rehearsal_representatives->index({index}).data_ptr(),
                m_num_bytes_per_representative,
                cudaMemcpyDefault,
                m_streams[3]
            ));
#else
            for (size_t r = 0; r < m_num_samples_per_activation; r++) {
                const int index = m_num_samples_per_activation * reprs_indices[i] + r;
                std::memcpy(
                    (char *) m_server_activations_mem[r].buffer->data_ptr() + m_num_bytes_per_activation * o,
                    m_rehearsal_activations->index({index}).data_ptr(),
                    m_num_bytes_per_activation
                );
            }
            // Copying the first representative only.
            const int index = m_num_samples_per_representative * reprs_indices[i];
            std::memcpy(
                (char *) m_server_activations_rep_mem[0].buffer->data_ptr() + m_num_bytes_per_representative * o,
                m_rehearsal_representatives->index({index}).data_ptr(),
                m_num_bytes_per_representative
            );
#endif
            o++;
        }
    }
    ASSERT(nactivations == o);
    ASSERT(samples.size() == attached_metadata.size());

    if (nactivations > 0) {
#ifndef WITHOUT_CUDA
        // The rehearsal_mutex is still held
        cudaStreamSynchronize(m_streams[2]);
#endif

        const auto num_bytes_representatives = nactivations * m_num_bytes_per_representative;
        const auto num_bytes_activations = nactivations * m_num_bytes_per_activation;

        for (size_t r = 0; r < m_num_samples_per_activation; r++) {
            m_server_activations_mem[r].bulk(0, num_bytes_activations)
                >> client_activations_bulks[r](offset * m_num_bytes_per_activation, num_bytes_activations).on(req.get_endpoint());
        }
        m_server_activations_rep_mem[0].bulk(0, num_bytes_representatives)
            >> client_activations_rep_bulks[0](offset * m_num_bytes_per_representative, num_bytes_representatives).on(req.get_endpoint());
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
void distributed_stream_loader_t::accumulate(
        const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations)
{
    nvtx3::scoped_range nvtx{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");
    if (!m_use_allocated_variables)
        throw std::runtime_error("You didn't pass variables to augment, so you should call use_these_allocated_variables() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);

    request_queue.emplace_back(queue_item_t(representatives, targets, activations));
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
void distributed_stream_loader_t::accumulate(
        const std::vector<torch::Tensor>& representatives, const torch::Tensor& targets, const std::vector<torch::Tensor>& activations,
        const std::vector<torch::Tensor>& aug_representatives, const torch::Tensor& aug_targets, const torch::Tensor& aug_weights,
        const std::vector<torch::Tensor>& buf_activations, const torch::Tensor& buf_activations_rep)
{
    nvtx3::scoped_range nvtx{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    auto item = queue_item_t(representatives, targets, activations, aug_representatives, aug_targets, aug_weights, buf_activations, buf_activations_rep);
    request_queue.emplace_back(item);
    lock.unlock();
    request_cond.notify_one();
}

/**
 * This function should be called when using the `flyweight` buffer implementation.
 */
void distributed_stream_loader_t::use_these_allocated_variables(
        const std::vector<torch::Tensor>& buf_representatives, const torch::Tensor& buf_targets, const torch::Tensor& buf_weights, const std::vector<torch::Tensor>& buf_activations, const torch::Tensor& buf_activations_rep)
{
    if (m_task_type == Task::REHEARSAL || m_task_type == Task::REHEARSAL_KD) {
        m_buf_representatives = std::make_shared<std::vector<torch::Tensor>>(buf_representatives);
        m_buf_targets = std::make_shared<torch::Tensor>(buf_targets);
        m_buf_weights = std::make_shared<torch::Tensor>(buf_weights);

        ASSERT(m_buf_representatives->size() == m_num_samples_per_representative);
        ASSERT(m_buf_representatives->at(0).dim() > 0 && m_buf_targets->dim() == 1);

        m_R = m_buf_representatives->at(0).sizes()[0];
        ASSERT(m_R > 0 && m_R == m_buf_targets->sizes()[0]
                    && m_R == m_buf_weights->sizes()[0]);
        for (size_t i = 1; i < m_num_samples_per_representative; i++)
            ASSERT(m_R == m_buf_representatives->at(i).sizes()[0]);
    }

    if (m_task_type == Task::KD || m_task_type == Task::REHEARSAL_KD) {
        m_buf_activations = std::make_shared<std::vector<torch::Tensor>>(buf_activations);
        m_buf_activations_rep = std::make_shared<torch::Tensor>(buf_activations_rep);

        ASSERT(m_buf_activations->size() == m_num_samples_per_activation);
    }

    m_use_allocated_variables = true;
}

/**
 * This is called from Python in a synchronous fashion. We consume the
 * data processed by the client thread. If no data is ready, we just wait,
 * blocking the Python thread.
 */
std::tuple<int, int> distributed_stream_loader_t::wait()
{
    nvtx3::scoped_range nvtx{"wait"};

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (response_queue.empty())
        request_cond.wait(lock);
    auto batch = response_queue.front();
    response_queue.pop_front();
    return std::make_tuple(batch.aug_size, batch.activations_size);
}

void distributed_stream_loader_t::enable_augmentation(bool state)
{
    m_augmentation_enabled = state;
}

void distributed_stream_loader_t::measure_performance(bool state)
{
    m_measure_performance = state;
}

size_t distributed_stream_loader_t::get_rehearsal_size()
{
    return m_rehearsal_size;
}

std::vector<float> distributed_stream_loader_t::get_metrics(size_t i_batch)
{
    if (!m_metrics.count(i_batch))
        return {};
    return m_metrics[i_batch].get_durations();
}

void distributed_stream_loader_t::finalize()
{
    std::unique_lock<tl::mutex> lock(request_mutex);
    request_queue.push_back(queue_item_t());
    lock.unlock();
    request_cond.notify_one();

    if (m_verbose)
        DBG("[" << m_provider_id << "] Finalize signal sent...");
}

distributed_stream_loader_t::~distributed_stream_loader_t() noexcept
{
    m_server_representatives_procedure.deregister();
    m_server_activations_procedure.deregister();

    // Pop the finalize callback. If this destructor was called
    // from the finalization callback, there is nothing to pop
    get_engine().pop_finalize_callback(this);

    es->join();

#ifndef WITHOUT_CUDA
    for (auto& stream : m_streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
#endif
}
