/* Compare independently compiled scalar and SSE2 kernels on synthetic vectors. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef int (*cosine_fn)(const unsigned char *, const unsigned char *, size_t, size_t, double *);
static cosine_fn load_fn(void *handle) {
    void *symbol = dlsym(handle, "similarity_cosine_f32le");
    cosine_fn fn = NULL;
    if (sizeof(fn) != sizeof(symbol) || !symbol)
        exit(2);
    memcpy(&fn, &symbol, sizeof(fn));
    return fn;
}
static void store(unsigned char *p, float x) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    for (int j = 0; j < 4; j++)
        p[j] = (unsigned char)(bits >> (8 * j));
}
int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    void *a = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL), *b = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!a || !b)
        return 2;
    cosine_fn scalar = load_fn(a), sse2 = load_fn(b);
    unsigned char left[768 * 4 + 1], right[768 * 4 + 1];
    double max_abs = 0, max_rel = 0;
    uint32_t state = 123456789u;
    for (size_t n = 1; n <= 768; n++) {
        for (size_t i = 0; i < n; i++) {
            state = 1664525u * state + 1013904223u;
            float x = (float)((int)(state % 20001) - 10000) / 10000.0f;
            state = 1664525u * state + 1013904223u;
            float y = (float)((int)(state % 20001) - 10000) / 10000.0f;
            store(left + 1 + 4 * i, x);
            store(right + 1 + 4 * i, y);
        }
        double x, y;
        if (!scalar(left + 1, right + 1, 4 * n, 4 * n, &x) ||
            !sse2(left + 1, right + 1, 4 * n, 4 * n, &y))
            return 1;
        double delta = fabs(x - y);
        if (delta > max_abs)
            max_abs = delta;
        if (fabs(x) > 1e-12 && delta / fabs(x) > max_rel)
            max_rel = delta / fabs(x);
        if (delta >= 1e-12)
            return 1;
    }
    printf("cases=768 scalar_vs_sse2_max_abs=%.17g max_relative_nonzero=%.17g tolerance=1e-12\n",
           max_abs, max_rel);
    dlclose(b);
    dlclose(a);
    return 0;
}
