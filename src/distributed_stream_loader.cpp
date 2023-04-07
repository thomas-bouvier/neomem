#include "distributed_stream_loader.hpp"
#include "mpi_utils.hpp"

#include <tuple>
#include <chrono>
#include <stdexcept>

#include <thallium/serialization/stl/tuple.hpp>
#include <thallium/serialization/stl/vector.hpp>
#include <cereal/types/string.hpp>

#ifndef WITHOUT_CUDA
#include <cuda_runtime.h>
#endif

#include "debug.hpp"

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
distributed_stream_loader_t::distributed_stream_loader_t(const engine_loader_t& _engine_loader,
    Task _task_type, unsigned int _K, unsigned int _N, unsigned int _R, unsigned int _C,
    int64_t seed, unsigned int _num_samples_per_representative,
    std::vector<long> _representative_shape,
    bool discover_endpoints, bool _verbose)
        : tl::provider<distributed_stream_loader_t>(_engine_loader.get_engine(), _engine_loader.get_id()),
        engine_loader(_engine_loader),
        task_type(_task_type), K(_K), N(_N), R(_R), C(_C), rand_gen(seed),
        num_samples_per_representative(_num_samples_per_representative),
        representative_shape(_representative_shape), verbose(_verbose) {
    num_bytes_per_representative = 4 * std::accumulate(representative_shape.begin(), representative_shape.end(), 1, std::multiplies<int>());

    init_rehearsal_buffers(true);

    define("get_samples", &distributed_stream_loader_t::get_remote_samples);
    // Register the remote procedure
    get_samples_procedure = get_engine().define("get_samples");

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
    for (auto endpoint : endpoints) {
        std::cout << "Looking up " << endpoint.first << ", " << endpoint.second << std::endl;
        tl::endpoint server = get_engine().lookup(endpoint.first);
        provider_handles.emplace_back(tl::provider_handle(server, endpoint.second));
    }
}

void distributed_stream_loader_t::init_rehearsal_buffers(bool pin_buffers) {
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
    rehearsal_metadata.insert(rehearsal_metadata.begin(), K, std::make_pair(0, 0.0));
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
    expose_memory(client_mem);

    while (true) {
        // WAITING for data
        std::unique_lock<tl::mutex> lock(request_mutex);
        while (request_queue.empty())
            request_cond.wait(lock);
        auto batch = request_queue.front();
        batch.aug_size = 0;
        request_queue.pop_front();
        lock.unlock();
        metrics[i_batch].accumulate_time = std::chrono::system_clock::now() - metrics[i_batch].last_accumulate_time;

        // COPY m into m'
        auto now = std::chrono::system_clock::now();
        // An empty batch is a signal for shutdown
        if (!batch.samples.defined())
            break;
        int batch_size = batch.samples.sizes()[0];
        ASSERT(batch.targets.dim() == 1 && batch_size == batch.targets.sizes()[0]);

        // Initialization of the augmented result
        if (!use_allocated_variables) {
            ASSERT(batch.aug_samples.dim() > 0 && batch.aug_targets.dim() == 1);
            auto actual_R = batch.aug_samples.sizes()[0] - batch_size;
            ASSERT(actual_R > 0 && actual_R + batch_size == batch.aug_targets.sizes()[0]
                && actual_R + batch_size == batch.aug_weights.sizes()[0]);
            copy_last_batch(batch, batch_size);
        }
        metrics[i_batch].batch_copy_time = std::chrono::system_clock::now() - now;

        if (augmentation_enabled)
            augment_batch(batch, batch_size);

        i_batch++;

        // UPDATE buffer
        now = std::chrono::system_clock::now();
        populate_rehearsal_buffer(batch);
        update_representative_weights(R, batch_size);
        metrics[i_batch].buffer_update_time = std::chrono::system_clock::now() - now;

        lock.lock();
        response_queue.emplace_back(batch);
        lock.unlock();
        request_cond.notify_one();
    }
}

void distributed_stream_loader_t::copy_last_batch(queue_item_t &batch, int batch_size) {
#ifndef WITHOUT_CUDA
    ASSERT(cudaMemcpy((char *) batch.aug_samples.data_ptr(),
                        batch.samples.data_ptr(),
                        batch_size * num_bytes_per_representative,
                        cudaMemcpyDeviceToDevice
    ) == cudaSuccess);
    ASSERT(cudaMemcpy((char *) batch.aug_targets.data_ptr(),
                        batch.targets.data_ptr(),
                        batch_size * batch.targets[0].nbytes(),
                        cudaMemcpyDeviceToDevice
    ) == cudaSuccess);
#else
    std::memcpy((char *) batch.aug_samples.data_ptr(),
                batch.samples.data_ptr(),
                batch_size * num_bytes_per_representative
    );
    std::memcpy((char *) batch.aug_targets.data_ptr(),
                batch.targets.data_ptr(),
                batch_size * batch.targets[0].nbytes()
    );
#endif

    for (int i = 0; i < batch_size; i++)
        batch.aug_weights.index_put_({i}, 1.0);

    batch.aug_size = batch_size;
}

/**
 * Should return a bulk (called only once)
 * - in the best case, to an once-allocated variable from Python
 + - 
 */
void distributed_stream_loader_t::expose_memory(exposed_memory_t &mem) {
    if (use_allocated_variables) {
        if (!engine_loader.is_cuda_rdma_enabled() && alloc_aug_samples.is_cuda())
            throw std::invalid_argument("The augmented mini-batch is stored in CUDA memory, allocated policy is selected, but cuda+verbs is not supported");

        mem.buffer = &alloc_aug_samples;
    } else {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
#ifndef WITHOUT_CUDA
        if (engine_loader.is_cuda_rdma_enabled())
            options = options.device(torch::kCUDA);
#endif
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
 *
 */
void distributed_stream_loader_t::augment_batch(queue_item_t &batch, int batch_size) {
    std::unordered_map<int, std::vector<int>> indices_per_node = pick_random_indices(R);

    // PREPARE bulk
    auto now = std::chrono::system_clock::now();
    // Iterate over nodes and issuing corresponding rpc requests
    auto j = 0;
    std::vector<tl::async_response> responses;
    for (const auto& indices : indices_per_node) {
        tl::provider_handle& ph = provider_handles[indices.first];
        auto response = get_samples_procedure.on(ph).async(client_mem.bulk, indices.second, j);
        responses.push_back(std::move(response));

        j += indices.second.size() * num_samples_per_representative * num_bytes_per_representative;
    }
    ASSERT(responses.size() == indices_per_node.size());
    metrics[i_batch].bulk_prepare_time = std::chrono::system_clock::now() - now;

    // SAMPLE globally
    now = std::chrono::system_clock::now();
    // Waiting for rpc requests to resolve
    for (size_t i = 0; i < indices_per_node.size(); i++) {
        std::vector<std::tuple<int, double, size_t>> metadata = responses[i].wait();

        for (const auto &it : metadata) {
            int label;
            double weight;
            size_t num_targets;
            std::tie(label, weight, num_targets) = it;
            for (size_t j = 0; j < num_targets; j++) {
                if (use_allocated_variables) {
                    alloc_aug_targets.index_put_({batch.aug_size}, label);
                    alloc_aug_weights.index_put_({batch.aug_size}, weight);
                } else {
                    batch.aug_targets.index_put_({batch.aug_size}, label);
                    batch.aug_weights.index_put_({batch.aug_size}, weight);
                }
                batch.aug_size++;
            }
        }
    }
    metrics[i_batch].rpcs_resolve_time = std::chrono::system_clock::now() - now;

    if (!use_allocated_variables)
        copy_exposed_buffer_to_aug_batch(batch, batch_size);
}

/**
 * We should copy from the exposed bulk to the minibatch
 */
void distributed_stream_loader_t::copy_exposed_buffer_to_aug_batch(queue_item_t &batch, int batch_size) {
    auto nbytes = num_samples_per_representative * num_bytes_per_representative;

    if (!use_allocated_variables) {
        // COPY representatives
        auto now = std::chrono::system_clock::now();
#ifndef WITHOUT_CUDA
        ASSERT(cudaMemcpy((char *) batch.aug_samples.data_ptr() + batch_size * nbytes,
                            client_mem.buffer->data_ptr(),
                            (batch.aug_size - batch_size) * nbytes,
                            cudaMemcpyHostToDevice
        ) == cudaSuccess);
#else
    std::memcpy((char *) batch.aug_samples.data_ptr() + batch_size * nbytes,
                client_mem.buffer->data_ptr(),
                (batch.aug_size - batch_size) * nbytes
    );
#endif
        metrics[i_batch].representatives_copy_time = std::chrono::system_clock::now() - now;
    }
}

/**
 * Selection without replacement from remote nodes + current node.
 *
 * The map returned by this function maps remote node indices to local indices.
 * Local indices might be used to access the provider_handles vector.
 */
std::unordered_map<int, std::vector<int>> distributed_stream_loader_t::pick_random_indices(int R) {
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
    std::unique_lock<tl::mutex> lock(rehearsal_mutex);

    auto batch_size = batch.samples.sizes()[0];
    std::uniform_int_distribution<unsigned int> dice(0, batch_size - 1);
    for (int i = 0; i < batch_size; i++) {
        if (dice(rand_gen) >= C)
            break;
        int label = 0;
        if (task_type == Classification)
            label = batch.targets[i].item<int>();

        size_t index = -1;
        if (rehearsal_metadata[label].first < N)
            index = rehearsal_metadata[label].first;
        else
            index = dice(rand_gen);
        // The random replacement strategy does nothing sometimes
        if (index < N) {
            for (size_t r = 0; r < num_samples_per_representative; r++) {
                //TODO reconstruction
                size_t j = N * label + index + r;
                ASSERT(j < K * N * num_samples_per_representative);
                rehearsal_tensor->index_put_({static_cast<int>(j)}, batch.samples.index({i}));
            }
            if (index >= rehearsal_metadata[label].first) {
                rehearsal_size++;
                rehearsal_metadata[label].first++;
            }
            history_count++;
        }
    }
}

void distributed_stream_loader_t::update_representative_weights(int num_representatives, int batch_size) {
    double weight = (double) batch_size / (double) (num_representatives * rehearsal_size);
    for (auto &pair : rehearsal_metadata) {
        pair.second = std::max(std::log(pair.first * weight), 1.0);
    }
}

void distributed_stream_loader_t::get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices, int offset) {
    int c = 0, o = 0;
    std::vector<std::tuple<size_t, double, std::vector<int>>> samples;

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
    if (rehearsal_size > 0) {
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

    std::vector<std::tuple<int, double, size_t>> metadata;
    for (const auto &el : samples) {
        int label;
        double weight;
        std::vector<int> reprs_indices;
        std::tie(label, weight, reprs_indices) = el;
        metadata.emplace_back(std::make_tuple(label, weight, reprs_indices.size()));

        for (size_t i = 0; i < reprs_indices.size(); i++) {
            server_mem.buffer->index_put_({o}, rehearsal_tensor->index({reprs_indices[i]}));
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
 */
void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets) {
    if (!started)
        throw std::runtime_error("Call start() before accumulate()");

    metrics[i_batch].last_accumulate_time = std::chrono::system_clock::now();
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
 */
void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    if (!started)
        throw std::runtime_error("Call start() before accumulate()");

    metrics[i_batch].last_accumulate_time = std::chrono::system_clock::now();
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets, aug_samples, aug_targets, aug_weights));
    lock.unlock();
    request_cond.notify_one();
}

void distributed_stream_loader_t::use_these_allocated_variables(const torch::Tensor &aug_samples,
                const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    alloc_aug_samples = aug_samples;
    alloc_aug_targets = aug_targets;
    alloc_aug_weights = aug_weights;
    use_allocated_variables = true;

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
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (response_queue.empty())
        request_cond.wait(lock);
    auto batch = response_queue.front();
    response_queue.pop_front();
    return batch.aug_size;
}

void distributed_stream_loader_t::enable_augmentation(bool state) {
    augmentation_enabled = state;
}

size_t distributed_stream_loader_t::get_rehearsal_size() {
    return rehearsal_size;
}

size_t distributed_stream_loader_t::get_history_count() {
    return history_count;
}

std::vector<double> distributed_stream_loader_t::get_metrics(size_t i_batch) {
    if (!metrics.count(i_batch))
        return {};
    return metrics[i_batch].get_durations();
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
