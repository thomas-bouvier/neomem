#ifndef __METRICS
#define __METRICS

#include <chrono>

struct metrics_t {
    std::chrono::duration<double> batch_copy_time;
    std::chrono::duration<double> bulk_prepare_time;
    std::chrono::duration<double> rpcs_resolve_time;
    std::chrono::duration<double> representatives_copy_time;
    std::chrono::duration<double> buffer_update_time;

    std::vector<double> get_durations() const {
        return {
            batch_copy_time.count(),
            bulk_prepare_time.count(),
            rpcs_resolve_time.count(),
            representatives_copy_time.count(),
            buffer_update_time.count()
        };
    }
};

#endif
