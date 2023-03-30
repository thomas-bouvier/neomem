#include "distributed_stream_loader.hpp"
#include "mpi_utils.hpp"

#include <tuple>
#include <chrono>

#include <thallium/serialization/stl/tuple.hpp>
#include <thallium/serialization/stl/vector.hpp>
#include <cereal/types/string.hpp>

#ifndef WITHOUT_CUDA
#include <cuda_runtime.h>
#endif

#define __DEBUG
#define __ASSERT
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
    Task _task_type, unsigned int _K, unsigned int _N, unsigned int _C,
    int64_t seed, unsigned int _num_samples_per_representative,
    std::vector<long> _representative_shape,
    bool discover_endpoints, bool _verbose)
        : tl::provider<distributed_stream_loader_t>(_engine_loader.get_engine(), _engine_loader.get_id()),
        engine_loader(_engine_loader),
        task_type(_task_type), K(_K), N(_N), C(_C), rand_gen(seed),
        num_samples_per_representative(_num_samples_per_representative),
        representative_shape(_representative_shape), verbose(_verbose) {
    define("get_samples", &distributed_stream_loader_t::get_remote_samples);
    // Register the remote procedure
    get_samples_procedure = get_engine().define("get_samples");

    // The thread executing the actual client issuing rpcs
    es = tl::xstream::create();
    async_thread = es->make_thread([this]() {
        async_process();
    });

    // If enabled, get the remote endpoints via the MPI publishing mechanism
    if (discover_endpoints) {
        std::map<std::string, int> all_endpoints = gather_endpoints();
        if (all_endpoints.size() > 0) {
            std::cout << "endpoint size " << all_endpoints.size() << std::endl;
            register_endpoints(all_endpoints);
        }
    }

    auto size = K * N * num_samples_per_representative;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
    rehearsal_vector.insert(rehearsal_vector.begin(), size, torch::zeros(representative_shape, options));
    rehearsal_metadata.insert(rehearsal_metadata.begin(), K, std::make_pair(0, 0.0));
    DBG("Distributed buffer memory allocated!");
}

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

/**
 * This method consumes the data pushed into the request_queue by accumulate(),
 * processes it, samples data from all other servers, and push the new data into
 * the response_queue (which will be consumed in turn by wait()).
 */
void distributed_stream_loader_t::async_process() {
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
        int R = 0;

        // Initialization of the augmented result
        if (use_allocated_variables) {
            ASSERT(alloc_aug_samples.dim() > 0 && alloc_aug_targets.dim() == 1);
            R = alloc_aug_samples.sizes()[0];
            ASSERT(R > 0 && R == alloc_aug_targets.sizes()[0]
                && R == alloc_aug_weights.sizes()[0]);
        } else {
            ASSERT(batch.aug_samples.dim() > 0 && batch.aug_targets.dim() == 1);
            R = batch.aug_samples.sizes()[0] - batch_size;
            ASSERT(R > 0 && R + batch_size == batch.aug_targets.sizes()[0]
                && R + batch_size == batch.aug_weights.sizes()[0]);
            copy_last_batch(batch, batch_size);
        }
        metrics[i_batch].batch_copy_time = std::chrono::system_clock::now() - now;

        if (augmentation_enabled)
            augment_batch(batch, R);

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
/*
#ifndef WITHOUT_CUDA
    ASSERT(cudaMemcpy((char *) batch.aug_samples.data_ptr(),
                        batch.samples.data_ptr(),
                        batch_size * batch.samples[0].nbytes(),
                        cudaMemcpyDeviceToDevice
    ) == cudaSuccess);
    ASSERT(cudaMemcpy((char *) batch.aug_targets.data_ptr(),
                        batch.targets.data_ptr(),
                        batch_size * batch.targets[0].nbytes(),
                        cudaMemcpyDeviceToDevice
    ) == cudaSuccess);
    for (int i = 0; i < batch_size; i++) {
        batch.aug_weights.index_put_({i}, 1.0);
    }
#elif
*/
    for (int i = 0; i < batch_size; i++) {
        batch.aug_samples.index_put_({i}, batch.samples[i]);
        batch.aug_targets.index_put_({i}, batch.targets[i]);
        batch.aug_weights.index_put_({i}, 1.0);
    }
//#endif
    batch.aug_size = batch_size;
}

int distributed_stream_loader_t::augment_batch(queue_item_t &batch, int R) {
    auto batch_size = batch.samples.sizes()[0];
    auto nbytes = batch.samples[0].nbytes();

    // PREPARE bulk
    auto now = std::chrono::system_clock::now();
    // R will be greater if last batch has a smaller size
    //TODO: could be simplified (indices generation)
    std::unordered_map<int, std::vector<int>> indices_per_node = pick_random_indices(R);

    int k = 0;
    torch::Tensor* buffer;
    auto shape = representative_shape;
    shape.insert(shape.begin(), R);
    bool cpu_buffer = false;
    if (use_allocated_variables) {
        if (!engine_loader.is_cuda_rdma_enabled() && alloc_aug_samples.is_cuda()) {
            buffer = new torch::Tensor(torch::zeros(shape));
            cpu_buffer = true;
        } else {
            buffer = &alloc_aug_samples;
        }
    } else {
        if (!engine_loader.is_cuda_rdma_enabled() && batch.aug_samples.is_cuda()) {
            buffer = new torch::Tensor(torch::zeros(shape));
            cpu_buffer = true;
        } else {
            buffer = &batch.aug_samples;
            k = batch_size;
        }
    }

    // These objects should live as long as rpc requests are not resolved
    std::vector<tl::async_response> responses;
    std::vector<std::vector<std::pair<void*, std::size_t>>> segments;
    std::vector<tl::bulk> bulks;

    struct hg_bulk_attr attr;
    memset(&attr, 0, sizeof(attr));
    if (buffer->is_cuda())
        attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_CUDA;
    else
        attr.mem_type = (hg_mem_type_t) HG_MEM_TYPE_HOST;

    // Iterate over nodes and issuing corresponding rpc requests
    int j = k;
    for (const auto& indices : indices_per_node) {
        auto& inserted_segments = segments.emplace_back(indices.second.size() * num_samples_per_representative);
        // Each segment maps to an individual tensor
        for (auto& s : inserted_segments) {
            ASSERT(buffer->is_contiguous());
            s.first = (char *) buffer->data_ptr() + j * nbytes;
            s.second = nbytes;
            j++;
        }

        tl::provider_handle& ph = provider_handles[indices.first];
        tl::bulk bulk = get_engine().expose(inserted_segments, tl::bulk_mode::write_only, attr);
        bulks.push_back(std::move(bulk));

        auto response = get_samples_procedure.on(ph).async(bulks.back(), indices.second);
        responses.push_back(std::move(response));
    }
    ASSERT(responses.size() == indices_per_node.size());
    metrics[i_batch].bulk_prepare_time = std::chrono::system_clock::now() - now;

    // SAMPLE globally
    now = std::chrono::system_clock::now();
    // Waiting for rpc requests to resolve
    for (size_t i = 0; i < indices_per_node.size(); i++) {
        decltype(responses.begin()) completed;
        std::vector<std::tuple<int, double, size_t>> metadata = tl::async_response::wait_any(responses.begin(), responses.end(), completed);
        responses.erase(completed);

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

    if (cpu_buffer) {
        // COPY representatives
        now = std::chrono::system_clock::now();
        ASSERT(k - batch_size <= R);
#ifndef WITHOUT_CUDA
        if (use_allocated_variables) {
            ASSERT(cudaMemcpy((char *) alloc_aug_samples.data_ptr() + k * nbytes,
                                buffer->data_ptr(),
                                batch.aug_size * nbytes,
                                cudaMemcpyHostToDevice
            ) == cudaSuccess);
        } else {
            ASSERT(cudaMemcpy((char *) batch.aug_samples.data_ptr() + batch_size * nbytes,
                                buffer->data_ptr(),
                                (batch.aug_size - batch_size) * nbytes,
                                cudaMemcpyHostToDevice
            ) == cudaSuccess);
        }
#endif
        delete buffer;
        metrics[i_batch].representatives_copy_time = std::chrono::system_clock::now() - now;
    }

    /*
    TEST:
    for (int i = 0; i < batch.aug_size; i++) {
        if (batch.aug_targets[i].item().toInt() != batch.aug_samples[i][0][0][0].item().toInt()) {
            std::cout << "fail it " << i << std::endl;
            std::cout << "label " << batch.aug_targets[i].item().toInt() << std::endl;
            std::cout << "value " << batch.aug_samples[i][0][0][0].item().toInt() << std::endl;
            ASSERT(false);
        }
    }
    */

    return k;
}

/**
 * Selection without replacement from remote nodes + current node.
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
    auto batch_size = batch.samples.sizes()[0];
    std::uniform_int_distribution<unsigned int> dice(0, batch_size - 1);
    for (int i = 0; i < batch_size; i++) {
        if (dice(rand_gen) >= C)
            break;
        int label = 0;
        if (task_type == Classification)
            label = batch.targets[i].item<int>();

        std::unique_lock<tl::mutex> lock(rehearsal_mutex);
        size_t index = -1;
        if (rehearsal_metadata[label].first < N)
            index = rehearsal_metadata[label].first;
        else
            index = dice(rand_gen);
        // The random replacement strategy does nothing sometimes
        if (index < N) {
            for (size_t r = 0; r < num_samples_per_representative; r++) {
                //TODO reconstruction
                torch::Tensor tensor = batch.samples.index({i}).detach().clone().to(torch::kCPU);
                ASSERT(tensor.nbytes() != 0);
                auto j = N * label + index + r;
                ASSERT(j < K * N * num_samples_per_representative);
                rehearsal_vector[j] = tensor;
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

void distributed_stream_loader_t::get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices) {
    size_t c = 0;
    rehearsal_map_t samples;

    /**
    * Input
    * Vector of indices
    *
    * Output
    * Rehearsal buffer, unordered map indexed by labels
    * - (label1, (weight, reprs))
    * - (label2, (weight, reprs))
    * If a representative is already present for a label, the representative is
    * appended to reprs.
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
            representative_t repr;
            for (size_t r = 0; r < num_samples_per_representative; r++) {
                auto tensor = rehearsal_vector[i * N + rehearsal_repr_of_class_index + r];
                /*
                TEST:
                ASSERT(tensor[0][0][0].item().toInt() == i);
                */
                repr.emplace_back(tensor);
            }

            auto weight = rehearsal_metadata[i].second;
            samples[i].first = weight;
            samples[i].second.emplace_back(repr);
            c++;
        }
    }

    if (verbose) {
        std::cout << "Sending " << c << "/" << indices.size()  << " representatives from "
            << samples.size() << " different classes to remote node (endpoint: "
            << req.get_endpoint() << ")" << std::endl;
    }

    /**
    * Fill the RDMA buffer with tensors
    *
    * Input
    * Rehearsal buffer, unordered map indexed by labels
    * - (label1, (weight, reprs))
    * - (label2, (weight, reprs))
    *
    *
    * Output
    * Metadata, vector (to preserve the order) of tuples
    * - {(label1, weight, num_reprs), (label2, weight, num_reprs)}
    * Segments
    * - {(ptrrepA, nbytesA) (ptrrepB, nbytesB) (ptrrepC, nbytesC) (ptrrepD, nbytesD)}
    *
    * repA and repB are of label1, repC and repD are of label2
    * TODO: complete this comment to explain how representatives are expanded
    **/
    std::vector<std::tuple<int, double, size_t>> metadata;
    std::vector<std::pair<void*, size_t>> segments;
    for (const auto &it : samples) {
        auto label = it.first;
        auto weight = it.second.first;
        const representative_collection_t& reprs = it.second.second;
        metadata.emplace_back(std::make_tuple(label, weight, reprs.size()));

        for (const representative_t& repr : reprs) {
            ASSERT(repr.size() == num_samples_per_representative);
            for (const torch::Tensor& tensor : repr) {
                ASSERT(tensor.nbytes() != 0);
                ASSERT(tensor.is_contiguous());
                /*
                TEST:
                ASSERT(label == tensor[0][0][0].item().toInt());
                DBG("value " << tensor[0][0][0].item().toInt());
                */
                segments.emplace_back(tensor.data_ptr(), tensor.nbytes());
            }
        }
    }
    ASSERT(c == segments.size());
    ASSERT(samples.size() == metadata.size());

    /*
    TEST:
    int s = 0;
    DBG("checking if the metadata reflects the segments, iterating on metadata (FAILING)..");
    for (auto const &it : metadata) {
        auto label = std::get<0>(it);
        DBG("for label " << label);
        for (int num_reps = 0; num_reps < std::get<2>(it); num_reps++) {
            DBG("num_reps " << num_reps);
            DBG("value " << torch::from_blob(segments[s].first, representative_shape, torch::kFloat32)[0][0][0].item().toInt());
            ASSERT(label == torch::from_blob(segments[s].first, representative_shape, torch::kFloat32)[0][0][0].item().toInt());
            s++;
        }
    }
    */

    if (segments.size() > 0) {
        tl::bulk bulk = get_engine().expose(segments, tl::bulk_mode::read_only);
        bulk >> b.on(req.get_endpoint());
    }
    req.respond(metadata);
}

/**
 * This is called from Python in a synchronous fashion. We push the incoming
 * data to the request_queue for it to be consumed by the client thread in an
 * asynchronous fashion.
 */
void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets) {
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
