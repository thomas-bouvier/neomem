#ifndef __METRICS
#define __METRICS

#include <chrono>

struct metrics_t {
    float batch_copy_time;
    float bulk_prepare_time;
    float rpcs_resolve_time;
    float representatives_copy_time;
    float queue_time;
    float buffer_update_time;

    std::vector<float> get_durations() const {
        return {
            batch_copy_time,
            bulk_prepare_time,
            rpcs_resolve_time,
            representatives_copy_time,
            buffer_update_time
        };
    }
};

#endif
