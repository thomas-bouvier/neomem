#include "distributed_stream_loader.hpp"

#include <tuple>
#include <assert.h>

#include <thallium/serialization/stl/tuple.hpp>
#include <thallium/serialization/stl/vector.hpp>
#include <cuda_runtime.h>
#include <cereal/types/string.hpp>

#include "mpi_utils.hpp"

#define __DEBUG
#define __ASSERT
#include "debug.hpp"

using namespace torch::indexing;

//TODO: constructor with device registration
engine_loader_t::engine_loader_t(const std::string &address, uint16_t provider_id) :
    server_engine(address, THALLIUM_SERVER_MODE, true, POOL_SIZE) {
    std::cout << "Server running at address " << server_engine.self()
                << " with provider id " << provider_id << std::endl;
}

engine_loader_t::~engine_loader_t() {
    server_engine.wait_for_finalize();
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
distributed_stream_loader_t::distributed_stream_loader_t(Task _task_type, unsigned int _K, unsigned int _N, unsigned int _C,
    int64_t seed, uint16_t _server_id, const std::string& server_address,
    unsigned int _num_samples_per_representative, std::vector<long> _representative_shape,
    bool _cuda_rdma, bool discover_endpoints, bool _verbose)
        : engine_loader_t(server_address, _server_id), tl::provider<distributed_stream_loader_t>(server_engine, _server_id),
        task_type(_task_type), K(_K), N(_N), C(_C), rand_gen(seed),
        num_samples_per_representative(_num_samples_per_representative),
        representative_shape(_representative_shape), cuda_rdma(_cuda_rdma),
        verbose(_verbose) {

    define("get_samples", &distributed_stream_loader_t::get_remote_samples);
    // Register the remote procedure
    get_samples_procedure = server_engine.define("get_samples");

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

    rehearsal_vector.insert(rehearsal_vector.begin(), K * N * num_samples_per_representative, torch::zeros(representative_shape));
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

    std::map<std::string, int> endpoints = {{server_engine.self(), server_id}};
    auto all_endpoints = gather_dictionary(endpoints, MAX_CF_LENGTH, num_workers, rank);

    if (!was_initialized) {
        MPI_Finalize();
    }
    return all_endpoints;
}

void distributed_stream_loader_t::register_endpoints(const std::map<std::string, int>& endpoints) {
    for (auto endpoint : endpoints) {
        std::cout << "Looking up " << endpoint.first << ", " << endpoint.second << std::endl;
        tl::endpoint server = server_engine.lookup(endpoint.first);
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
        std::unique_lock<tl::mutex> lock(request_mutex);
        while (request_queue.empty())
            request_cond.wait(lock);
        auto batch = request_queue.front();
        request_queue.pop_front();
        lock.unlock();

        // An empty batch is a signal for shutdown
        if (!batch.samples.defined())
            break;
        int batch_size = batch.samples.sizes()[0];
        assert(batch.targets.dim() == 1 && batch_size == batch.targets.sizes()[0]
            && batch.aug_samples.dim() > 0 && batch.aug_targets.dim() == 1);
        int R = batch.aug_samples.sizes()[0] - batch_size;
        assert(R > 0 && R + batch_size == batch.aug_targets.sizes()[0]
            && R + batch_size == batch.aug_weights.sizes()[0]);

        // Initialization of the augmented result
        for (int i = 0; i < batch_size; i++) {
            batch.aug_samples.index_put_({i}, batch.samples[i]);
            batch.aug_targets.index_put_({i}, batch.targets[i]);
            batch.aug_weights.index_put_({i}, 1.0);
            batch.aug_size = batch_size;
        }

        if (augmentation_enabled)
            augment_batch(batch, R);

        lock.lock();
        response_queue.emplace_back(batch);
        lock.unlock();
        request_cond.notify_one();

        populate_rehearsal_buffer(batch);
        update_representative_weights(R, batch_size);
    }
}

int distributed_stream_loader_t::augment_batch(queue_item_t &batch, int R) {
    auto batch_size = batch.samples.sizes()[0];
    auto nbytes = batch.samples[0].nbytes();

    // R will be greater if last batch has a smaller size
    //TODO: could be simplified (indices generation)
    std::unordered_map<int, std::vector<int>> indices_per_node = pick_random_indices(R);

    int k;
    torch::Tensor* buffer;
    bool use_cpu_buffer = !cuda_rdma && batch.aug_samples.device().type() == torch::kCUDA;
    if (use_cpu_buffer) {
        // Use a temporary CPU buffer that will be copied to CUDA later
        auto shape = representative_shape;
        shape.insert(shape.begin(), R);
        buffer = new torch::Tensor(torch::zeros(shape));
        k = 0;
    } else {
        buffer = &batch.aug_samples;
        k = batch_size;
    }

    // Iterate over nodes
    std::vector<tl::async_response> responses;
    std::vector<std::vector<std::pair<void*, std::size_t>>> all_segments;
    for (const auto& indices : indices_per_node) {
        auto& segments = all_segments.emplace_back(indices.second.size() * num_samples_per_representative);
        // Each segment maps to an individual tensor
        for (auto& segment : segments) {
            ASSERT(buffer->is_contiguous());
            segment.first = (char *) buffer->data_ptr() + k * nbytes;
            segment.second = nbytes;
            k++;
        }

        tl::provider_handle& ph = provider_handles[indices.first];
        tl::bulk local_bulk = server_engine.expose(segments, tl::bulk_mode::write_only);
        responses.emplace_back(get_samples_procedure.on(ph).async(local_bulk, indices.second));
    }

    k = batch_size;
    for (int i = 0; i < indices_per_node.size(); i++) {
        std::vector<std::tuple<int, double, std::size_t>> metadata = responses[i].wait();
        for (const auto &it : metadata) {
            int label;
            double weight;
            std::size_t num_targets;
            std::tie(label, weight, num_targets) = it;
            for (int j = 0; j < num_targets; j++) {
                batch.aug_targets.index_put_({k}, label);
                batch.aug_weights.index_put_({k}, weight);
                k++;
            }
        }
    }
    batch.aug_size = k;

    if (use_cpu_buffer) {
        ASSERT(k - batch_size <= R);
        ASSERT(cudaMemcpy((char *) batch.aug_samples.data_ptr() + batch_size * nbytes,
                            buffer->data_ptr(),
                            (k - batch_size) * nbytes, 
                            cudaMemcpyHostToDevice
        ) == cudaSuccess);
        delete buffer;
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
 * Selection without replacement from remote nodes + current node
 * get_samples in Python
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

/**
 * Accumulate in Python
 * Tip: pick indices then replace
 */
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
        int index = -1;
        if (rehearsal_metadata[label].first < N)
            index = rehearsal_metadata[label].first;
        else
            index = dice(rand_gen);
        // The random replacement strategy does nothing sometimes
        if (index < N) {
            for (int r = 0; r < num_samples_per_representative; r++) {
                //TODO reconstruction
                torch::Tensor tensor = batch.samples.index({i}).detach().clone();
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

/**
 * Tip: keep references in temporary data structure
 * Tip: no need to lock, tensors are thread safe and we only care about having
 * valid data, no matter the data
 */
void distributed_stream_loader_t::get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices) {
    int c = 0;
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

            int j = -1, i = 0;
            for (; i < rehearsal_metadata.size(); i++) {
                if (rehearsal_metadata[i].first == 0)
                    continue;
                j++;
                if (j == rehearsal_class_index)
                    break;
            }

            const size_t rehearsal_repr_of_class_index = (index % N) % rehearsal_metadata[i].first;
            representative_t repr;
            for (int r = 0; r < num_samples_per_representative; r++) {
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
    std::vector<std::tuple<int, double, std::size_t>> metadata;
    std::vector<std::pair<void*, std::size_t>> segments;
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
void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples.to(torch::kCPU), targets.to(torch::kCPU),
            aug_samples, aug_targets, aug_weights));
    lock.unlock();
    request_cond.notify_one();
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
