/* sampling_strategies.cpp -- Synthetic behavior checks for native samplers. */
#include "llama.h"

#include <cassert>
#include <cmath>

static bool HasToken(const llama_token_data_array &candidates, llama_token token) {
    for (size_t i = 0; i < candidates.size; ++i) {
        if (candidates.data[i].id == token) return true;
    }
    return false;
}

int main() {
    {
        llama_token_data values[] = {{0, 3.0f, 0.0f}, {1, 2.9f, 0.0f},
            {2, 0.0f, 0.0f}, {3, -4.0f, 0.0f}};
        llama_token_data_array candidates = {values, 4, -1, false};
        llama_sampler *typical = llama_sampler_init_typical(0.5f, 1);
        assert(typical != nullptr);
        llama_sampler_apply(typical, &candidates);
        assert(candidates.size > 0 && candidates.size < 4);
        assert(HasToken(candidates, 0));
        llama_sampler_free(typical);
    }

    {
        llama_token_data values_a[] = {{0, 3.0f, 0.0f}, {1, 2.9f, 0.0f},
            {2, 0.0f, 0.0f}};
        llama_token_data values_b[] = {{0, 3.0f, 0.0f}, {1, 2.9f, 0.0f},
            {2, 0.0f, 0.0f}};
        llama_token_data_array a = {values_a, 3, -1, false};
        llama_token_data_array b = {values_b, 3, -1, false};
        llama_sampler *xtc = llama_sampler_init_xtc(1.0f, 0.4f, 1, 42);
        assert(xtc != nullptr);
        llama_sampler *clone = llama_sampler_clone(xtc);
        assert(xtc != nullptr && clone != nullptr);
        llama_sampler_apply(xtc, &a);
        llama_sampler_apply(clone, &b);
        assert(a.size == 2 && b.size == 2);
        assert(!HasToken(a, 0) && !HasToken(b, 0));
        assert(a.data[0].id == b.data[0].id && a.data[1].id == b.data[1].id);
        llama_sampler_reset(xtc);
        llama_sampler_reset(clone);
        llama_sampler_free(xtc);
        llama_sampler_free(clone);
    }

    {
        llama_token_data values[] = {{1, 10.0f, 0.0f}, {2, 10.0f, 0.0f},
            {3, 10.0f, 0.0f}};
        llama_token_data_array candidates = {values, 3, -1, false};
        llama_sampler *dry = llama_sampler_init_dry(nullptr, 1.0f, 1.75f,
            1, 16, nullptr, 0);
        assert(dry != nullptr);
        llama_sampler_accept(dry, 1);
        llama_sampler_accept(dry, 2);
        llama_sampler_accept(dry, 1);
        llama_sampler_apply(dry, &candidates);
        assert(values[1].logit < 10.0f);
        assert(values[2].logit == 10.0f);
        llama_sampler *clone = llama_sampler_clone(dry);
        assert(clone != nullptr);
        llama_sampler_reset(dry);
        llama_sampler_reset(clone);
        llama_sampler_free(dry);
        llama_sampler_free(clone);
    }

    return 0;
}
