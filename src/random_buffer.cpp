#include "random_buffer.hpp"

#include "debug.hpp"
#include "memory_utils.hpp"
#include <nvtx3/nvtx3.hpp>

/**
 * Initializes the rehearsal buffers for representatives.
 */
RandomBuffer::RandomBuffer(RehearsalConfig config)
    : m_config(config)
    , m_rand_gen(config.seed)
{
    if (m_config.task_type == Task::REHEARSAL || m_config.task_type == Task::REHEARSAL_KD) {
        allocate(
            m_rehearsal_representatives,
            m_config.num_samples_per_representative,
            m_config.representative_shape,
            torch::cuda::is_available()
        );
        initialize_num_bytes_per_representative();
    }

    if (m_config.task_type == Task::KD || m_config.task_type == Task::REHEARSAL_KD) {
        allocate(
            m_rehearsal_activations,
            m_config.num_samples_per_activation,
            m_config.activation_shape,
            torch::cuda::is_available()
        );
        initialize_num_bytes_per_activation();
    }
}

/**
 * Initialize
 */
void RandomBuffer::allocate(
        std::unique_ptr<torch::Tensor>& storage, size_t nsamples, std::vector<long> sample_shape, bool pin_buffers)
{
    nvtx3::scoped_range nvtx{"init_rehearsal_buffer"};

    auto size = m_config.K * m_config.N * nsamples;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pin_buffers);

    sample_shape.insert(sample_shape.begin(), size);
    storage = std::make_unique<torch::Tensor>(torch::empty(sample_shape, options));
    ASSERT(storage->is_contiguous());
    rehearsal_metadata.insert(rehearsal_metadata.begin(), m_config.K, std::make_pair(0, 1.0));
    rehearsal_counts.insert(rehearsal_counts.begin(), m_config.K, 0);
}

/**
 * Calculates the number of bytes per representative based on the shape.
 */
void RandomBuffer::initialize_num_bytes_per_representative() {
    m_num_bytes_per_representative = 4 * std::accumulate(m_config.representative_shape.begin(), m_config.representative_shape.end(), 1, std::multiplies<int>());
}

void RandomBuffer::initialize_num_bytes_per_activation() {
    m_num_bytes_per_activation = 2 * std::accumulate(m_config.activation_shape.begin(), m_config.activation_shape.end(), 1, std::multiplies<int>());
    if (!m_config.half_precision) {
        m_num_bytes_per_activation *= 2;
    }
}

/*
 * Sample C random elements from the given batch to populate the rehearsal
 * buffer.
 */
void RandomBuffer::populate(const queue_item_t& batch, unsigned int nelements)
{
    nvtx3::scoped_range nvtx{"populate_rehearsal_buffer"};

    std::uniform_int_distribution<unsigned int> dice_candidate(0, batch.get_size() - 1);
    std::uniform_int_distribution<unsigned int> dice_buffer(0, m_config.N - 1);
    for (size_t i = 0; i < batch.get_size(); i++) {
        //if (dice(m_rand_gen) >= C)
        //    break;
        int label = (m_config.K == 1) ? 0 : batch.m_targets[i].item<int>();

        size_t index = -1;
        if (rehearsal_metadata[label].first < m_config.N) {
            index = rehearsal_metadata[label].first;
        } else {
            if (dice_candidate(m_rand_gen) >= nelements)
                continue;
            index = dice_buffer(m_rand_gen);
        }

        size_t j = m_config.N * label + index;
        ASSERT(j < m_config.K * m_config.N);
#ifndef WITHOUT_CUDA
        cudaStream_t stream = m_streams[1];
#else
        NullStream stream;
#endif

        for (size_t k = 0; k < m_config.num_samples_per_representative; k++) {
            smart_copy(
                (char *) m_rehearsal_representatives->data_ptr() + m_num_bytes_per_representative * (m_config.num_samples_per_representative * j + k),
                (char *) batch.m_representatives[k].data_ptr() + m_num_bytes_per_representative * i,
                m_num_bytes_per_representative,
                stream
            );
        }
        if (m_config.task_type == Task::KD || m_config.task_type == Task::REHEARSAL_KD) {
            for (size_t k = 0; k < m_config.num_samples_per_activation; k++) {
                smart_copy(
                    (char *) m_rehearsal_activations->data_ptr() + m_num_bytes_per_activation * (m_config.num_samples_per_activation * j + k),
                    (char *) batch.m_activations[k].data_ptr() + m_num_bytes_per_activation * i,
                    m_num_bytes_per_activation,
                    stream
                );
            }
        }

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
}

/**
 * With big datasets like ImageNet, the following formula results in really
 * small weights. Keeping this function as future work.
 */
void RandomBuffer::update_representative_weights(const queue_item_t& batch, int num_representatives)
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
std::vector<std::tuple<size_t, float, std::vector<int>>> RandomBuffer::get_indices(const std::vector<int>& indices) const {
    std::vector<std::tuple<size_t, float, std::vector<int>>> samples;

    if (m_rehearsal_size > 0) {
        for (auto index : indices) {
            size_t rehearsal_class_index = index / m_config.N;
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

            const size_t rehearsal_repr_of_class_index = (index % m_config.N) % rehearsal_metadata[i].first;

            if (std::none_of(samples.begin(), samples.end(), [&](const auto& el) { return std::get<0>(el) == i; })) {
                samples.emplace_back(i, rehearsal_metadata[i].second, std::vector<int>{});
            }
            for (auto& el : samples) {
                if (std::get<0>(el) == i) {
                    std::get<2>(el).push_back(i * m_config.N + rehearsal_repr_of_class_index);
                }
            }
        }
    }

    return samples;
}

size_t RandomBuffer::get_rehearsal_size()
{
    return m_rehearsal_size;
}
