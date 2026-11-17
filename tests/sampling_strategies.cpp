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
        // The pinned engine defines sigma <= 0 as a no-op. A positive value
        // truncates candidates while remaining a filter (not a selector).
        llama_token_data values[] = {{0, 4.0f, 0.0f}, {1, 3.0f, 0.0f},
            {2, 2.0f, 0.0f}, {3, 0.0f, 0.0f}};
        llama_token_data_array candidates = {values, 4, -1, false};
        llama_sampler *disabled = llama_sampler_init_top_n_sigma(0.0f);
        assert(disabled != nullptr);
        llama_sampler_apply(disabled, &candidates);
        assert(candidates.size == 4);
        llama_sampler_free(disabled);

        llama_sampler *sigma = llama_sampler_init_top_n_sigma(0.5f);
        assert(sigma != nullptr);
        llama_sampler *sigma_clone = llama_sampler_clone(sigma);
        assert(sigma_clone != nullptr);
        llama_sampler_apply(sigma, &candidates);
        size_t finite_logits = 0;
        for (size_t i = 0; i < candidates.size; ++i) {
            if (std::isfinite(candidates.data[i].logit)) ++finite_logits;
        }
        assert(finite_logits > 0 && finite_logits < 4);
        llama_sampler_free(sigma_clone);
        llama_sampler_free(sigma);
    }

    {
        // delta == 0 follows the same native scaling as the legacy sampler.
        llama_token_data legacy_values[] = {{0, 3.0f, 0.0f}, {1, 1.0f, 0.0f}};
        llama_token_data extended_values[] = {{0, 3.0f, 0.0f}, {1, 1.0f, 0.0f}};
        llama_token_data_array legacy = {legacy_values, 2, -1, false};
        llama_token_data_array extended = {extended_values, 2, -1, false};
        llama_sampler *standard = llama_sampler_init_temp(0.7f);
        llama_sampler *dynamic_disabled = llama_sampler_init_temp_ext(0.7f, 0.0f, 1.0f);
        assert(standard != nullptr && dynamic_disabled != nullptr);
        llama_sampler_apply(standard, &legacy);
        llama_sampler_apply(dynamic_disabled, &extended);
        assert(std::fabs(legacy_values[0].logit - extended_values[0].logit) < 1e-6f);
        assert(std::fabs(legacy_values[1].logit - extended_values[1].logit) < 1e-6f);
        llama_sampler_free(dynamic_disabled);
        llama_sampler_free(standard);
    }

    {
        // Exercise construction, clone and destruction for the two additional
        // public samplers. Adaptive-P is terminal and selects an index.
        llama_token_data values[] = {{0, 4.0f, 0.0f}, {1, 3.0f, 0.0f},
            {2, 2.0f, 0.0f}};
        llama_token_data_array candidates = {values, 3, -1, false};
        llama_sampler *temperature = llama_sampler_init_temp_ext(0.8f, 0.2f, 1.0f);
        llama_sampler *adaptive = llama_sampler_init_adaptive_p(0.3f, 0.9f, 17);
        assert(temperature != nullptr && adaptive != nullptr);
        llama_sampler *temperature_clone = llama_sampler_clone(temperature);
        llama_sampler *adaptive_clone = llama_sampler_clone(adaptive);
        assert(temperature_clone != nullptr && adaptive_clone != nullptr);
        llama_sampler_apply(temperature, &candidates);
        llama_sampler_apply(adaptive, &candidates);
        assert(candidates.selected >= 0 && candidates.selected < (int)candidates.size);
        llama_sampler_accept(adaptive, candidates.data[candidates.selected].id);
        llama_sampler_free(adaptive_clone);
        llama_sampler_free(temperature_clone);
        llama_sampler_free(adaptive);
        llama_sampler_free(temperature);
    }

    {
        // Characterize the pinned Adaptive-P clone implementation. It copies
        // EMA/pending fields but re-seeds rather than copying the advanced RNG.
        llama_sampler *adaptive = llama_sampler_init_adaptive_p(0.3f, 0.9f, 29);
        assert(adaptive != nullptr);
        for (int i = 0; i < 5; ++i) {
            llama_token_data values[] = {{0, 4.0f, 0.0f}, {1, 3.0f, 0.0f},
                {2, 2.0f, 0.0f}, {3, 1.0f, 0.0f}};
            llama_token_data_array candidates = {values, 4, -1, false};
            llama_sampler_apply(adaptive, &candidates);
            assert(candidates.selected >= 0 && candidates.selected < 4);
            llama_sampler_accept(adaptive, candidates.data[candidates.selected].id);
        }
        llama_sampler *clone = llama_sampler_clone(adaptive);
        assert(clone != nullptr);
        bool diverged = false;
        for (int i = 0; i < 16 && !diverged; ++i) {
            llama_token_data left[] = {{0, 4.0f, 0.0f}, {1, 3.0f, 0.0f},
                {2, 2.0f, 0.0f}, {3, 1.0f, 0.0f}};
            llama_token_data right[] = {{0, 4.0f, 0.0f}, {1, 3.0f, 0.0f},
                {2, 2.0f, 0.0f}, {3, 1.0f, 0.0f}};
            llama_token_data_array a = {left, 4, -1, false};
            llama_token_data_array b = {right, 4, -1, false};
            llama_sampler_apply(adaptive, &a);
            llama_sampler_apply(clone, &b);
            assert(a.selected >= 0 && a.selected < 4);
            assert(b.selected >= 0 && b.selected < 4);
            diverged = left[a.selected].id != right[b.selected].id;
            llama_sampler_accept(adaptive, left[a.selected].id);
            llama_sampler_accept(clone, right[b.selected].id);
        }
        assert(diverged);
        llama_sampler_free(clone);
        llama_sampler_free(adaptive);
    }

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
