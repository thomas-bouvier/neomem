#include "distributed_stream_loader.hpp"
#include "mpi_utils.hpp"
#include "debug.hpp"
#include "timer.hpp"

#include <tuple>
#include <chrono>
#include <stdexcept>

#include <cereal/types/string.hpp>
#include <nvtx3/nvtx3.hpp>
#include <thallium/serialization/stl/tuple.hpp>
#include <thallium/serialization/stl/vector.hpp>


using namespace torch::indexing;

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
        engine_loader(_engine_loader),
        task_type(_task_type), K(_K), N(_N), R(_R), C(_C), rand_gen(seed),
        num_samples_per_representative(_num_samples_per_representative),
        representative_shape(_representative_shape), buffer_strategy(_buffer_strategy), verbose(_verbose) {
    num_bytes_per_representative = 4 * std::accumulate(representative_shape.begin(), representative_shape.end(), 1, std::multiplies<int>());

#ifndef WITHOUT_CUDA
    init_rehearsal_buffers(true);
        auto device = cuda::device::current::get();
        std::generate_n(
            // first stream for client, second for server
            std::back_inserter(m_streams), 2,
            [&device]() {
                return device.create_stream(cuda::stream::sync);
            }
        );
#else
    init_rehearsal_buffers(false);
#endif

    m_server_procedure = define("get_samples", &distributed_stream_loader_t::get_remote_samples);
    // Register the remote procedure
    m_client_procedure = get_engine().define("get_samples");

    // If enabled, get the remote endpoints via the MPI publishing mechanism
    if (discover_endpoints) {
        std::map<std::string, int> all_endpoints = gather_endpoints();
        if (all_endpoints.size() > 0) {
            std::cout << "endpoint size " << all_endpoints.size() << std::endl;
            register_endpoints(all_endpoints);
        }
    }
}

/**
 * Contact all other nodes to get their endpoints. The returned dictionary maps
 * endpoints (keys) to provider ids (values).
 */
std::map<std::string, int> distributed_stream_loader_t::gather_endpoints() const {
    nvtx3::scoped_range r{"gather_endpoints"};

    int rank, num_workers = 0;
    // MPI has maybe been initialized by horovodrun
    int mpi_initialized = true;
    bool was_initialized = true;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(NULL, NULL);
        was_initialized = false;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers);

    std::map<std::string, int> endpoints = {{get_engine().self(), engine_loader.get_id()}};
    auto all_endpoints = gather_dictionary(endpoints, num_workers);

    if (!was_initialized) {
        MPI_Finalize();
    }
    return all_endpoints;
}

void distributed_stream_loader_t::register_endpoints(const std::map<std::string, int>& endpoints) {
    nvtx3::scoped_range r{"register_endpoints"};

    for (auto endpoint : endpoints) {
        std::cout << "Looking up " << endpoint.first << ", " << endpoint.second << std::endl;
        tl::endpoint server = get_engine().lookup(endpoint.first);
        provider_handles.emplace_back(tl::provider_handle(server, endpoint.second));
    }
}

void distributed_stream_loader_t::init_rehearsal_buffers(bool pin_buffers) {
    nvtx3::scoped_range r{"init_rehearsal_buffer"};

    auto size = K * N * num_samples_per_representative;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

#ifndef WITHOUT_CUDA
    options = options.pinned_memory(pin_buffers);
#else
    if (pin_buffers)
        throw std::invalid_argument("Pinning the rehearsal buffer requires CUDA");
#endif

    auto rehearsal_shape = representative_shape;
    rehearsal_shape.insert(rehearsal_shape.begin(), size);
    rehearsal_tensor = new torch::Tensor(torch::empty(rehearsal_shape, options));
    ASSERT(rehearsal_tensor->is_contiguous());
    rehearsal_metadata.insert(rehearsal_metadata.begin(), K, std::make_pair(0, 1.0));
    rehearsal_counts.insert(rehearsal_counts.begin(), K, 0);
    DBG("Distributed buffer memory allocated!");

    auto shape = representative_shape;
    shape.insert(shape.begin(), R);
    server_mem.buffer = new torch::Tensor(torch::empty(shape, options));
    server_mem.segments.emplace_back(server_mem.buffer->data_ptr(), R * num_samples_per_representative * num_bytes_per_representative);
    server_mem.bulk = get_engine().expose(server_mem.segments, tl::bulk_mode::read_only);
}

/**
 *
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
 */
void distributed_stream_loader_t::async_process() {
    init_receiving_rdma_buffer(client_mem);

    while (true) {
        nvtx3::mark("while start in async_process");

        // WAITING for data
        std::unique_lock<tl::mutex> lock(request_mutex);
        while (request_queue.empty())
            request_cond.wait(lock);
        auto batch = request_queue.front();
        batch.aug_size = 0;
        request_queue.pop_front();
        lock.unlock();

        nvtx3::mark("new iteration in async_process");

        // An empty batch is a signal for shutdown
        if (!batch.samples.defined())
            break;
        int batch_size = batch.samples.sizes()[0];
        ASSERT(batch.targets.dim() == 1 && batch_size == batch.targets.sizes()[0]);

        // Initialization of the augmented result
        if (!m_use_allocated_variables) {
            dest_samples = &batch.aug_samples;
            dest_targets = &batch.aug_targets;
            dest_weights = &batch.aug_weights;

            batch.m_augmentation_mark = batch_size;

            ASSERT(batch.aug_samples.dim() > 0 && batch.aug_targets.dim() == 1);
            auto actual_R = batch.aug_samples.sizes()[0] - batch_size;
            ASSERT(actual_R > 0 && actual_R + batch_size == batch.aug_targets.sizes()[0]
                && actual_R + batch_size == batch.aug_weights.sizes()[0]);

            copy_last_batch(batch, batch_size);
        } else {
            dest_samples = &alloc_aug_samples;
            dest_targets = &alloc_aug_targets;
            dest_weights = &alloc_aug_weights;
        }

        if (m_augmentation_enabled) {
            augment_batch(batch, batch_size);
        }

        populate_rehearsal_buffer(batch);
        //update_representative_weights(R, batch_size);

        i_batch++;

        lock.lock();
        response_queue.emplace_back(batch);
        lock.unlock();
        request_cond.notify_one();
    }
}

/**
 * Copy incoming sample/target pairs and associated weights to the next
 * augmented minibatch.
 */
void distributed_stream_loader_t::copy_last_batch(queue_item_t &batch, int batch_size) {
    nvtx3::scoped_range r{"copy_last_batch"};
    Timer timer(m_measure_performance);
#ifndef WITHOUT_CUDA
    timer.setStream(&m_streams[0]);
#endif
    timer.start();

    // Copy incoming samples into the next augmented minibatch
#ifndef WITHOUT_CUDA
    ASSERT(cudaMemcpyAsync(
        (char *) batch.aug_samples.data_ptr(),
        batch.samples.data_ptr(),
        batch_size * num_bytes_per_representative,
        cudaMemcpyDefault,
        m_streams[0].handle()
    ) == cudaSuccess);
    ASSERT(cudaMemcpyAsync(
        (char *) batch.aug_targets.data_ptr(),
        batch.targets.data_ptr(),
        batch_size * batch.targets.element_size(),
        cudaMemcpyDefault,
        m_streams[0].handle()
    ) == cudaSuccess);
#else
    std::memcpy(
        (char *) batch.aug_samples.data_ptr(),
        batch.samples.data_ptr(),
        batch_size * num_bytes_per_representative
    );
    std::memcpy(
        (char *) batch.aug_targets.data_ptr(),
        batch.targets.data_ptr(),
        batch_size * batch.targets.element_size()
    );
#endif

    // Initialize weights to 1.0
    torch::Tensor weights = torch::ones({batch_size}, batch.aug_weights.options());
#ifndef WITHOUT_CUDA
    ASSERT(cudaMemcpyAsync(
        (char *) batch.aug_weights.data_ptr(),
        weights.data_ptr(),
        batch_size * weights.element_size(),
        cudaMemcpyDefault,
        m_streams[0].handle()
    ) == cudaSuccess);
#else
    std::memcpy(
        (char *) batch.aug_weights.data_ptr(),
        weights.data_ptr(),
        batch_size * weights.element_size()
    );
#endif

    batch.aug_size = batch_size;

    m_metrics[i_batch].batch_copy_time = timer.end();
}

/**
 * Should return a bulk, taking into account:
 * - the 'allocated policy'
 * - the buffer strategy, NoBuffer, CPUBuffer or CUDABuffer
 */
void distributed_stream_loader_t::init_receiving_rdma_buffer(exposed_memory_t &mem) {
    nvtx3::scoped_range r{"init_receiving_rdma_buffer"};

    if (buffer_strategy == NoBuffer) {
        if (!m_use_allocated_variables)
            throw std::invalid_argument("NoBuffer policy is selected, so we should write in a variable declared on the Python side, which you didn't provide (or use CPUBuffer or CUDABuffer)");
        if (!engine_loader.is_cuda_rdma_enabled() && alloc_aug_samples.is_cuda())
            throw std::invalid_argument("NoBuffer policy is selected, but cuda+verbs is not supported");

        mem.buffer = &alloc_aug_samples;
    } else {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        if (buffer_strategy == CUDABuffer) {
            if (!engine_loader.is_cuda_rdma_enabled())
                throw std::invalid_argument("CUDABuffer policy is selected, but cuda+verbs is not supported");

            options = options.device(torch::kCUDA);
        } else {
#ifndef WITHOUT_CUDA
            options = options.pinned_memory(true);
#endif
        }

        auto shape = representative_shape;
        shape.insert(shape.begin(), R);
        mem.buffer = new torch::Tensor(torch::ones(shape, options));
    }

    struct hg_bulk_attr attr;
    memset(&attr, 0, sizeof(attr));
    if (mem.buffer->is_cuda())
        attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_CUDA;
    else
        attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_HOST;

    mem.segments.emplace_back(mem.buffer->data_ptr(), R * num_samples_per_representative * num_bytes_per_representative);
    mem.bulk = get_engine().expose(mem.segments, tl::bulk_mode::write_only, attr);
}

/**
 * This functions orchastrates the minibatch augmentation process, by
 * performing the following steps:
 *
 * 1) Dispatch rpcs to other processes. This is required by global sampling: a
 * subset of remote representatives are sampled to add diversity to the
 * augmented minibatch being constructed.
 * 2) Wait for rpcs to resolve. Depending if use_these_allocated_variables()
 * has been called before, remote data will be written in the `batch.aug_samples`
 * or `alloc_aug_samples`.
 */
void distributed_stream_loader_t::augment_batch(queue_item_t &batch, int batch_size) {
    nvtx3::scoped_range r{"augment_batch"};

    // Dispatch rpcs to other processes
    std::vector<tl::async_response> responses;
    dispatch_rpcs(responses);
    resolve_rpcs(responses, batch);

    // Copy representatives
    if (buffer_strategy != NoBuffer) {
        copy_exposed_buffer_to_aug_batch(batch, batch_size);
    }
}

/**
 *
 */
std::size_t distributed_stream_loader_t::dispatch_rpcs(std::vector<tl::async_response> &responses) {
    Timer timer(m_measure_performance);
#ifndef WITHOUT_CUDA
    timer.setStream(&m_streams[0]);
#endif
    timer.start();

    // Iterate over nodes and issuing corresponding rpc requests
    std::unordered_map<int, std::vector<int>> indices_per_node = pick_random_indices(R);
    auto j = 0;
    for (const auto& indices : indices_per_node) {
        tl::provider_handle& ph = provider_handles[indices.first];
        auto response = m_client_procedure.on(ph).async(client_mem.bulk, indices.second, j);
        responses.push_back(std::move(response));

        j += indices.second.size() * num_samples_per_representative * num_bytes_per_representative;
    }
    ASSERT(responses.size() == indices_per_node.size());

    m_metrics[i_batch].bulk_prepare_time = timer.end();
    return indices_per_node.size();
}

/**
 * Wait for rpc requests to resolve. The returned data is written in a buffer
 * representing the minibatch to augment.
 */
void distributed_stream_loader_t::resolve_rpcs(std::vector<tl::async_response>& responses, queue_item_t &batch) {
    Timer timer(m_measure_performance);
#ifndef WITHOUT_CUDA
    timer.setStream(&m_streams[0]);
#endif
    timer.start();

    for (size_t i = 0; i < responses.size(); i++) {
        std::vector<std::tuple<int, float, size_t>> metadata = responses[i].wait();

        for (const auto &it : metadata) {
            int label;
            float weight;
            size_t num_targets;
            std::tie(label, weight, num_targets) = it;

            for (size_t j = 0; j < num_targets; j++) {
#ifndef WITHOUT_CUDA
                torch::Tensor t_label = torch::tensor(label, dest_targets->options());
                torch::Tensor t_weight = torch::tensor(weight, dest_weights->options());
                // calculated pointer: dest_targets->data_ptr<long int>() + batch.aug_size
                // actual pointer: (*dest_targets)[batch.aug_size].data_ptr()
                cuda::memory::async::copy(
                    // no * batch.element_size() as type is given
                    dest_targets->data_ptr<long int>() + batch.aug_size,
                    t_label.data_ptr(),
                    t_label.element_size(),
                    m_streams[0]
                );
                cuda::memory::async::copy(
                    dest_weights->data_ptr<float>() + batch.aug_size,
                    t_weight.data_ptr(),
                    t_weight.element_size(),
                    m_streams[0]
                );
#else
                dest_targets->index_put_({batch.aug_size}, label);
                dest_weights->index_put_({batch.aug_size}, weight);
#endif
                batch.aug_size++;
            }
        }
    }

    m_metrics[i_batch].rpcs_resolve_time = timer.end();
}

/**
 * We should copy from the exposed bulk to the minibatch `dest_samples`. The
 * latter has either been passed during the last iteration (`batch.aug_samples`)
 * or has been allocated once at the beginning of the execution
 * (`alloc_aug_samples`).
 */
void distributed_stream_loader_t::copy_exposed_buffer_to_aug_batch(queue_item_t &batch, int batch_size) {
    nvtx3::scoped_range r{"copy_exposed_buffer_to_aug_batch"};
    Timer timer(m_measure_performance);
#ifndef WITHOUT_CUDA
    timer.setStream(&m_streams[0]);
#endif
    timer.start();

    auto nbytes = num_samples_per_representative * num_bytes_per_representative;
    size_t count = (batch.aug_size - batch.m_augmentation_mark) * nbytes;

#ifndef WITHOUT_CUDA
    ASSERT(cudaMemcpyAsync(
        (char *) dest_samples->data_ptr() + batch.m_augmentation_mark * nbytes,
        client_mem.buffer->data_ptr(),
        count,
        cudaMemcpyDefault,
        m_streams[0].handle()
    ) == cudaSuccess);
#else
    std::memcpy(
        (char *) dest_samples->data_ptr() + batch.m_augmentation_mark * nbytes,
        client_mem.buffer->data_ptr(),
        count
    );
#endif

    m_metrics[i_batch].representatives_copy_time = timer.end();
}

/**
 * Selection without replacement from remote nodes + current node.
 *
 * The map returned by this function maps remote node indices to local indices.
 * Local indices might be used to access the provider_handles vector.
 */
std::unordered_map<int, std::vector<int>> distributed_stream_loader_t::pick_random_indices(int R) {
    nvtx3::scoped_range r{"pick_random_indices"};

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

void distributed_stream_loader_t::populate_rehearsal_buffer(const queue_item_t& batch) {
    nvtx3::scoped_range r{"populate_rehearsal_buffer"};
    Timer timer(m_measure_performance);
#ifndef WITHOUT_CUDA
    timer.setStream(&m_streams[0]);
#endif
    timer.start();

    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    auto batch_size = batch.samples.sizes()[0];
    std::uniform_int_distribution<unsigned int> dice_candidate(0, batch_size - 1);
    std::uniform_int_distribution<unsigned int> dice_buffer(0, N - 1);
    for (int i = 0; i < batch_size; i++) {
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

#ifndef WITHOUT_CUDA
        size_t j = N * label + index;
        ASSERT(j < K * N * num_samples_per_representative);
        ASSERT(cudaMemcpyAsync(
            (char *) rehearsal_tensor->data_ptr() + num_bytes_per_representative * j,
            (char *) batch.samples.data_ptr() + num_bytes_per_representative * i,
            num_samples_per_representative * num_bytes_per_representative,
            cudaMemcpyDefault,
            m_streams[0].handle()
        ) == cudaSuccess);
#else
        for (size_t r = 0; r < num_samples_per_representative; r++) {
            size_t j = N * label + index + r;
            ASSERT(j < K * N * num_samples_per_representative);
            rehearsal_tensor->index_put_({static_cast<int>(j)}, batch.samples.index({i}));
        }
#endif
        if (index >= rehearsal_metadata[label].first) {
            m_rehearsal_size++;
            rehearsal_metadata[label].first++;
        }
        rehearsal_counts[label]++;
    }

    m_metrics[i_batch].buffer_update_time = timer.end();
}

/**
 * With big datasets like ImageNet, the following formula results in really
 * small weights. Keeping this function as future work.
 */
void distributed_stream_loader_t::update_representative_weights(int num_representatives, int batch_size) {
    nvtx3::scoped_range r{"update_representative_weights"};

    float weight = (float) batch_size / (float) (num_representatives * m_rehearsal_size);
    for (size_t i = 0; i < rehearsal_metadata.size(); i++) {
        rehearsal_metadata[i].second = std::max(std::log(rehearsal_counts[i] * weight), 1.0f);
    }
}

void distributed_stream_loader_t::get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices, int offset) {
    nvtx3::scoped_range r{"get_remote_samples"};

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

    if (verbose) {
        std::cout << "Sending " << c << "/" << indices.size()  << " representatives from "
            << samples.size() << " different classes to remote node (endpoint: "
            << req.get_endpoint() << ", writing at offset " << offset << ")" << std::endl;
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

    std::vector<std::tuple<int, float, size_t>> metadata;
    for (const auto &el : samples) {
        int label;
        float weight;
        std::vector<int> reprs_indices;
        std::tie(label, weight, reprs_indices) = el;

        metadata.emplace_back(std::make_tuple(label, weight, reprs_indices.size()));

        for (size_t i = 0; i < reprs_indices.size(); i++) {
#ifndef WITHOUT_CUDA
            ASSERT(cudaMemcpyAsync(
                (char *) server_mem.buffer->data_ptr() + num_bytes_per_representative * o,
                rehearsal_tensor->index({reprs_indices[i]}).data_ptr(),
                num_samples_per_representative * num_bytes_per_representative,
                cudaMemcpyDefault,
                m_streams[1].handle()
            ) == cudaSuccess);
#else
            server_mem.buffer->index_put_({o}, rehearsal_tensor->index({reprs_indices[i]}));
#endif
            o++;
        }
    }
    ASSERT(c == o);
    ASSERT(samples.size() == metadata.size());

    if (c > 0) {
        auto size = c * num_samples_per_representative * num_bytes_per_representative;
        server_mem.bulk(0, size) >> b(offset, size).on(req.get_endpoint());
    }

    req.respond(metadata);
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
    nvtx3::scoped_range r{"accumulate"};

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
    nvtx3::scoped_range r{"accumulate"};

    if (!started)
        throw std::runtime_error("Call start() before accumulate()");

    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets, aug_samples, aug_targets, aug_weights));
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
    m_use_allocated_variables = true;

    ASSERT(alloc_aug_samples.dim() > 0 && alloc_aug_targets.dim() == 1);
    R = alloc_aug_samples.sizes()[0];
    ASSERT(R > 0 && R == alloc_aug_targets.sizes()[0]
        && R == alloc_aug_weights.sizes()[0]);
}

/**
 * This is called from Python in a synchronous fashion. We consume the
 * data processed by the client thread. If no data is ready, we just wait,
 * blocking the Python thread.
 */
int distributed_stream_loader_t::wait() {
    nvtx3::scoped_range r{"wait"};

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

distributed_stream_loader_t::~distributed_stream_loader_t() {
    std::unique_lock<tl::mutex> lock(request_mutex);
    request_queue.push_back(queue_item_t());
    lock.unlock();
    request_cond.notify_one();

    get_engine().wait_for_finalize();
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
