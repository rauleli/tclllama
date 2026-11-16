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

static float score_token(llama_sampler * sampler, llama_token token) {
    llama_token_data candidate = {token, 2.0f, 0.0f};
    llama_token_data_array array = {&candidate, 1, -1, false};
    llama_sampler_apply(sampler, &array);
    return candidate.logit;
}

int main() {
    llama_sampler * active = llama_sampler_init_penalties(16, 8, 2.0f, 0.0f, 0.0f);
    if (!active) return 1;
    llama_sampler_accept(active, 7);
    llama_sampler * cloned = llama_sampler_clone(active);
    llama_sampler * fresh = llama_sampler_init_penalties(16, 8, 2.0f, 0.0f, 0.0f);
    llama_sampler * peer = cloned ? llama_sampler_clone(cloned) : nullptr;
    if (!cloned || !fresh || !peer) return 2;
    float before = score(active);
    float copied = score(cloned);
    float rebuilt = score(fresh);
    llama_sampler_accept(cloned, 8);
    float active_after_clone_change = score_token(active, 8);
    float clone_after_change = score_token(cloned, 8);
    float peer_after_clone_change = score_token(peer, 8);
    llama_sampler_reset(active);
    float cleared = score(active);
    std::printf("accepted=%.1f cloned=%.1f rebuilt=%.1f reset=%.1f clone_isolated=%.1f/%.1f/%.1f\n",
                before, copied, rebuilt, cleared, active_after_clone_change,
                clone_after_change, peer_after_clone_change);
    llama_sampler_free(active);
    llama_sampler_free(cloned);
    llama_sampler_free(fresh);
    llama_sampler_free(peer);
    return !(std::fabs(before - 1.0f) < 0.001f && std::fabs(copied - 1.0f) < 0.001f &&
             std::fabs(rebuilt - 2.0f) < 0.001f && std::fabs(cleared - 2.0f) < 0.001f &&
             std::fabs(active_after_clone_change - 2.0f) < 0.001f &&
             std::fabs(clone_after_change - 1.0f) < 0.001f &&
             std::fabs(peer_after_clone_change - 2.0f) < 0.001f);
}
