/* sampler_history.cpp -- Characterize native llama.cpp penalty history.
 * Build against the same static PIC archives as Makefile.local.
 * This isolated test uses synthetic logits and does not load a model.
 */
#include "llama.h"
#include <cmath>
#include <cstdio>

static float score(llama_sampler * sampler) {
    llama_token_data candidates[] = {{7, 2.0f, 0.0f}, {8, 1.5f, 0.0f}};
    llama_token_data_array array = {candidates, 2, -1, false};
    llama_sampler_apply(sampler, &array);
    return candidates[0].logit;
}

int main() {
    llama_sampler * active = llama_sampler_init_penalties(16, 8, 2.0f, 0.0f, 0.0f);
    if (!active) return 1;
    llama_sampler_accept(active, 7);
    llama_sampler * cloned = llama_sampler_clone(active);
    llama_sampler * fresh = llama_sampler_init_penalties(16, 8, 2.0f, 0.0f, 0.0f);
    if (!cloned || !fresh) return 2;
    float before = score(active);
    float copied = score(cloned);
    float rebuilt = score(fresh);
    llama_sampler_reset(active);
    float cleared = score(active);
    std::printf("accepted=%.1f cloned=%.1f rebuilt=%.1f reset=%.1f\n", before, copied, rebuilt, cleared);
    llama_sampler_free(active);
    llama_sampler_free(cloned);
    llama_sampler_free(fresh);
    return !(std::fabs(before - 1.0f) < 0.001f && std::fabs(copied - 1.0f) < 0.001f &&
             std::fabs(rebuilt - 2.0f) < 0.001f && std::fabs(cleared - 2.0f) < 0.001f);
}
