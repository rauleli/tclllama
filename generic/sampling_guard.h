/* sampling_guard.h -- Validate candidates before accepting a sampled token. */
#ifndef TCLLLAMA_SAMPLING_GUARD_H
#define TCLLLAMA_SAMPLING_GUARD_H
#include <cmath>
#include "llama.h"
static bool HasSelectableCandidate(const llama_token_data_array &candidates) {
    for (size_t i = 0; i < candidates.size; ++i) {
        if (std::isfinite(candidates.data[i].logit)) return true;
    }
    return false;
}
static bool HasValidSelection(const llama_token_data_array &candidates) {
    return candidates.selected >= 0 && (size_t)candidates.selected < candidates.size &&
        std::isfinite(candidates.data[candidates.selected].logit) &&
        std::isfinite(candidates.data[candidates.selected].p);
}
#endif
