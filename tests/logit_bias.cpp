/* logit_bias.cpp -- Synthetic checks of llama.cpp's public logit-bias sampler. */
#include "llama.h"
#include "../generic/sampling_guard.h"

#include <cassert>
#include <cmath>
#include <limits>

int main() {
    llama_token_data values[] = {
        {0, 1.0f, 0.0f},
        {1, 2.0f, 0.0f},
        {2, 3.0f, 0.0f}
    };
    llama_token_data_array candidates = {values, 3, -1, false};
    llama_logit_bias adjustments[] = {{0, 2.5f}, {1, -1.25f},
        {2, -std::numeric_limits<float>::infinity()}};
    llama_sampler *bias = llama_sampler_init_logit_bias(3, 3, adjustments);
    assert(bias != nullptr);

    llama_sampler_apply(bias, &candidates);
    assert(values[0].logit == 3.5f);
    assert(values[1].logit == 0.75f);
    assert(std::isinf(values[2].logit) && values[2].logit < 0.0f);
    assert(HasSelectableCandidate(candidates));

    llama_sampler_free(bias);
    llama_logit_bias suppress_all[] = {
        {0, -std::numeric_limits<float>::infinity()},
        {1, -std::numeric_limits<float>::infinity()},
        {2, -std::numeric_limits<float>::infinity()}
    };
    bias = llama_sampler_init_logit_bias(3, 3, suppress_all);
    assert(bias != nullptr);
    candidates.selected = -1;
    llama_sampler_apply(bias, &candidates);
    assert(!HasSelectableCandidate(candidates));
    llama_sampler_free(bias);
    return 0;
}
