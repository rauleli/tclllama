/* sampling_guard.cpp -- Validate grammar candidate checks without a model. */
#include "../generic/sampling_guard.h"
#include <cassert>
#include <limits>
int main() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    llama_token_data values[] = {{0, -inf, 0}, {1, -inf, nan}};
    llama_token_data_array a = {values, 2, 1, false};
    assert(!HasSelectableCandidate(a));
    assert(!HasValidSelection(a));
    values[0].logit = 1.0f;
    assert(HasSelectableCandidate(a));
    assert(!HasValidSelection(a));
    a.selected = 0;
    assert(HasValidSelection(a));
    values[0].p = nan;
    assert(!HasValidSelection(a));
    return 0;
}
