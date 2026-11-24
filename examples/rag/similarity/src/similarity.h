#ifndef TCLEMBEDDING_SIMILARITY_H
#define TCLEMBEDDING_SIMILARITY_H
#include <stddef.h>

/* IEEE 754 binary32 little-endian bytes; nonzero return means success.
 * Both arrays must be non-null, equally sized, nonempty, and a multiple of 4;
 * components and the result must be finite. Cosine also requires nonzero norms.
 * On failure the output is not modified. The caller owns all memory. */
int similarity_dot_f32le(const unsigned char *a, const unsigned char *b, size_t bytes_a,
                         size_t bytes_b, double *out);
int similarity_cosine_f32le(const unsigned char *a, const unsigned char *b, size_t bytes_a,
                            size_t bytes_b, double *out);
#endif
