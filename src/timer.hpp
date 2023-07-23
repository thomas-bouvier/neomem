#ifndef __TIMER
#define __TIMER

#include <chrono>

#ifndef WITHOUT_CUDA
#include "../third_party/cuda-api-wrappers/src/cuda/api.hpp"
#endif

class Timer {
public:
#ifndef WITHOUT_CUDA
    Timer(cuda::device_t& device, bool enabled = true) : m_events(
        device.create_event(
            cuda::event::sync_by_blocking,
            cuda::event::do_record_timings,
            cuda::event::not_interprocess
        ),
        device.create_event(
            cuda::event::sync_by_blocking,
            cuda::event::do_record_timings,
            cuda::event::not_interprocess
        )
    ), m_enabled(enabled) {}
#else
    Timer(bool enabled = true) : m_enabled(enabled) {}
#endif

#ifndef WITHOUT_CUDA
    void setStream(cuda::stream_t* stream) {
        m_stream = stream;
    }
#endif

    void start() {
        if (m_enabled) {
#ifndef WITHOUT_CUDA
            if (m_stream)
                m_stream->enqueue.event(m_events.first);
#endif
        }
    }

    float end() {
        if (m_enabled) {
#ifndef WITHOUT_CUDA
            if (m_stream) {
                m_stream->enqueue.event(m_events.second);
                m_stream->synchronize();
                return cuda::event::time_elapsed_between(m_events).count();
            }
            return 0;
#endif
        }

        return 0;
    }

private:
#ifndef WITHOUT_CUDA
    cuda::stream_t* m_stream = nullptr;
    std::pair<cuda::event_t, cuda::event_t> m_events;
#endif
    bool m_enabled;
};

#endif
