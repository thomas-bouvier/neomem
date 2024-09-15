#include "memory_utils.hpp"
#include <algorithm>

std::vector<std::pair<int, int>> merge_contiguous_memory(std::vector<std::pair<int, int>>& sections)
{
    std::vector<std::pair<int, int>> mergedPairs;
    if (sections.empty()) {
        return mergedPairs;
    }

    // Sort the sections based on the memory offset (first element of the pair)
    std::sort(sections.begin(), sections.end());

    // Merge contiguous chunks of memory
    mergedPairs.push_back(sections[0]);
    for (size_t i = 1; i < sections.size(); ++i) {
        int prevEnd = mergedPairs.back().first + mergedPairs.back().second;
        if (sections[i].first == prevEnd) {
            // Merge the current pair with the previous pair
            mergedPairs.back().second += sections[i].second;
        } else {
            // Non-contiguous, add as a new pair
            mergedPairs.push_back(sections[i]);
        }
    }

    return mergedPairs;
}