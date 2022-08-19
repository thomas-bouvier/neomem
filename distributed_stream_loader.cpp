#include "distributed_stream_loader.hpp"

#include <unordered_set>
#include <assert.h>

#include <thallium/serialization/stl/unordered_map.hpp>
#include <thallium/serialization/stl/pair.hpp>
#include <thallium/serialization/stl/vector.hpp>
#include <cereal/types/string.hpp>

#define __DEBUG
#include "debug.hpp"

using namespace torch::indexing;


distributed_stream_loader_t::distributed_stream_loader_t(Task _task_type, unsigned int _K, unsigned int _N, unsigned int _C,
    int64_t seed, uint16_t server_id, const std::string& server_address,
    std::vector<std::pair<int, std::string>>& endpoints, unsigned int _num_samples_per_representative, std::vector<long> _representative_shape)
        : tl::provider<distributed_stream_loader_t>([](uint16_t provider_id, const std::string& address) -> tl::engine& {
            static tl::engine myServer(address, THALLIUM_SERVER_MODE);
            std::cout << "Server running at address " << myServer.self()
                << " with provider id " << provider_id << std::endl;
            return myServer;
        }(server_id, server_address), server_id), task_type(_task_type), K(_K), N(_N),
        C(_C), rand_gen(seed), num_samples_per_representative(_num_samples_per_representative), representative_shape(_representative_shape) {
    define("get_samples", &distributed_stream_loader_t::get_remote_samples);
    get_samples_procedure = get_engine().define("get_samples");

    tl::managed<tl::xstream> es = tl::xstream::create();
    xstream = std::move(es);
    tl::managed<tl::thread> thread = xstream->make_thread([this]() {
        async_process();
    });
    async_thread = std::move(thread);

    endpoints.push_back(std::make_pair(server_id, server_address));
    if (endpoints.size() > 0) {
        std::cout << "endpoint size " << endpoints.size() << std::endl;
        add_endpoints(endpoints);
    }
}

void distributed_stream_loader_t::add_endpoints(const std::vector<std::pair<int, std::string>>& endpoints) {
    for (auto endpoint : endpoints) {
        std::cout << "Looking up " << endpoint.second << ", " << endpoint.first << std::endl;
        tl::endpoint server = get_engine().lookup(endpoint.second);
        provider_handles.emplace_back(tl::provider_handle(server, endpoint.first));
    }
}

void distributed_stream_loader_t::async_process() {
    while (true) {
        std::unique_lock<tl::mutex> lock(request_mutex);
        while (request_queue.empty())
            request_cond.wait(lock);
        auto batch = request_queue.front();
        request_queue.pop_front();
        lock.unlock();

        // an empty batch is a signal for shutdown
        if (!batch.samples.defined())
            break;
        int batch_size = batch.samples.sizes()[0];
        assert(batch.targets.dim() == 1 && batch_size == batch.targets.sizes()[0]
            && batch.aug_samples.dim() > 0 && batch.aug_targets.dim() == 1);
        int R = batch.aug_samples.sizes()[0] - batch_size;
        assert(R > 0 && R + batch_size == batch.aug_targets.sizes()[0]
            && R + batch_size == batch.aug_weights.sizes()[0]);

        // initialization of the augmented result
        for (int i = 0; i < batch_size; i++) {
            batch.aug_samples.index_put_({i}, batch.samples[i]);
            batch.aug_targets.index_put_({i}, batch.targets[i]);
            batch.aug_weights.index_put_({i}, 1.0);
        }

        auto indices_per_node = pick_random_indices(R);
        int j = batch_size;
        for (const auto& indices : indices_per_node) {
            // how many tensors returned by the current node?
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            std::vector<torch::Tensor> tensors(indices.second.size() * num_samples_per_representative, torch::zeros(representative_shape, options));
            for ([[maybe_unused]] const auto& tensor : tensors)
                assert(tensor.is_contiguous());
            std::vector<std::pair<void*, std::size_t>> segments(indices.second.size() * num_samples_per_representative);
            int i = 0;
            for (auto& rdma_tensor : segments) {
                rdma_tensor.first = tensors[i].data_ptr();
                rdma_tensor.second = tensors[i].nbytes();
                i++;
            }

            tl::provider_handle& ph = provider_handles[indices.first];
            tl::bulk local_bulk = get_engine().expose(segments, tl::bulk_mode::write_only);
            std::map<int, std::pair<int, int>> metadata = get_samples_procedure.on(ph)(local_bulk, indices.second);

            // received RDMA bulk, convert it back to Tensor now :)
            int t = 0;
            for (auto it = metadata.begin(); it != metadata.end(); it++) {
                int num_targets = it->second.first;
                for (int i = 0; i < num_targets; i++) {
                    batch.aug_samples.index_put_({j}, tensors[t]);
                    batch.aug_weights.index_put_({j}, it->second.second);
                    if (task_type == Classification) {
                        batch.aug_targets.index_put_({j}, it->first);
                        t++;
                    } else {
                        batch.aug_targets.index_put_({j}, tensors[t + 1]);
                        t += num_samples_per_representative;
                    }
                    j++;
                }
            }
        }
        batch.aug_size = j;

        lock.lock();
        response_queue.emplace_back(batch);
        lock.unlock();
        request_cond.notify_one();

        populate_rehearsal_buffer(batch, batch_size);
        update_representative_weights(R, batch_size);
    }
}

/**
 * Selection without replacement from remote nodes + current node
 * get_samples in Python
 */
std::unordered_map<int, std::vector<int>> distributed_stream_loader_t::pick_random_indices(int effective_representatives) {
    const unsigned int max_global_index = provider_handles.size() * K * N;
    std::uniform_int_distribution<unsigned int> dice(0, max_global_index - 1);
    std::vector<unsigned int> choices(effective_representatives);
    int i = 0;
    while (i < effective_representatives) {
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
        assert(node >= 0 && node < provider_handles.size());
        if (indices_per_node.find(node) == indices_per_node.end())
            indices_per_node.emplace(node, std::vector<int>());
        indices_per_node[node].push_back(local_index);
    }

    return indices_per_node;
}

/**
 * Accumulate in Python
 */
void distributed_stream_loader_t::populate_rehearsal_buffer(const queue_item_t& batch, int batch_size) {
    std::uniform_int_distribution<unsigned int> dice(0, batch_size - 1);
    for (int i = 0; i < batch_size; i++) {
        if (dice(rand_gen) >= C)
            break;
        int label = 0;
        if (task_type == Classification)
            label = batch.targets[i].item<int>();
        buffer_t& buffer = rehearsal_map[label].second;
        representative_t repr;
        repr.emplace_back(batch.samples.index({i}));
        if (task_type == Reconstruction)
            repr.emplace_back(batch.targets.index({i}));
        if (buffer.size() < N) {
            buffer.emplace_back(repr);
            rehearsal_size++;
        } else {
            unsigned int index = dice(rand_gen);
            if (index < N)
                buffer[index] = repr;
        }
        counts[label]++;
        history_count++;
    }
}

void distributed_stream_loader_t::update_representative_weights(int effective_representatives, int batch_size) {
    double weight = (double) batch_size / (double) (effective_representatives * rehearsal_size);
    for (auto& map_it : rehearsal_map) {
        map_it.second.first = std::max(std::log(counts[map_it.first] * weight), 1.0);
    }
}

void distributed_stream_loader_t::get_remote_samples(const tl::request& req, tl::bulk& b, const std::vector<int>& indices) const {
    rehearsal_map_t samples;
    if (rehearsal_size > 0) {
        for (auto index : indices) {
            size_t rehearsal_class = index / N;
            auto map_it = rehearsal_map.begin();
            std::advance(map_it, rehearsal_class % rehearsal_map.size());
            size_t rehearsal_class_index = (index % N) % map_it->second.second.size();
            assert(rehearsal_class_index < map_it->second.second.size());
            auto repr = map_it->second.second[rehearsal_class_index];
            auto label = map_it->first;
            auto weight = map_it->second.first;
            if (samples.find(label) == samples.end())
                samples.emplace(label, std::make_pair(weight, buffer_t()));
            samples[label].second.push_back(repr);
        }
    }

    std::cout << "Sending " << samples.size() << " representatives ("
        << indices.size() << " requested) to remote node (endpoint: "
        << req.get_endpoint() << ")" << std::endl;

    // Fill the RDMA buffer with tensors, ordering them by label
    int i = 0;
    std::map<int, std::pair<int, int>> metadata;
    std::vector<std::pair<void*, std::size_t>> segments(indices.size() * num_samples_per_representative);
    for (auto it = samples.begin(); it != samples.end(); it++) {
        buffer_t reprs = it->second.second;
        auto label = it->first;
        auto weight = it->second.first;
        metadata.insert({label, {reprs.size(), weight}});
        for (auto repr : reprs) {
            for (const auto& tensor : repr) {
                //TODO: use RDMA from GPU directly...
                auto contiguous_tensor = tensor.to(torch::kCPU).contiguous();
                assert(contiguous_tensor.is_contiguous());
                segments[i].first = contiguous_tensor.data_ptr();
                segments[i].second = contiguous_tensor.nbytes();
                i++;
            }
        }
    }
    for (size_t j = i; j < segments.size(); j++) {
        segments[j].first = nullptr;
        segments[j].second = 0;
    }

    tl::bulk bulk = get_engine().expose(segments, tl::bulk_mode::read_only);
    bulk >> b.on(req.get_endpoint());
    req.respond(metadata);
}

void distributed_stream_loader_t::accumulate(const torch::Tensor &samples, const torch::Tensor &targets,
                 const torch::Tensor &aug_samples, const torch::Tensor &aug_targets, const torch::Tensor &aug_weights) {
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (request_queue.size() == MAX_QUEUE_SIZE)
        request_cond.wait(lock);
    request_queue.emplace_back(queue_item_t(samples, targets, aug_samples, aug_targets, aug_weights));
    lock.unlock();
    request_cond.notify_one();
}

int distributed_stream_loader_t::wait() {
    std::unique_lock<tl::mutex> lock(request_mutex);
    while (response_queue.empty())
        request_cond.wait(lock);
    auto batch = response_queue.front();
    response_queue.pop_front();
    return batch.aug_size;
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
    async_thread->join();

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
