#ifndef __METRICS
#define __METRICS

#include <chrono>

struct metrics_t {
    std::chrono::system_clock::time_point last_accumulate_time;
    std::chrono::system_clock::time_point last_queue_time;

    std::chrono::duration<double> accumulate_time;
    std::chrono::duration<double> batch_copy_time;
    std::chrono::duration<double> bulk_prepare_time;
    std::chrono::duration<double> rpcs_resolve_time;
    std::chrono::duration<double> representatives_copy_time;
    std::chrono::duration<double> queue_time;
    std::chrono::duration<double> buffer_update_time;

    std::vector<double> get_durations() const {
        return {
            accumulate_time.count(),
            batch_copy_time.count(),
            bulk_prepare_time.count(),
            rpcs_resolve_time.count(),
            representatives_copy_time.count(),
            buffer_update_time.count()
        };
    }
};

#endif
