/* The configured x86-64 similarity objects must remain at the ABI baseline. */
#if defined(__x86_64__) &&                                                                         \
    (defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__) ||      \
     defined(__SSE4A__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) ||        \
     defined(__FMA__) || defined(__FMA4__) || defined(__F16C__) || defined(__BMI__) ||             \
     defined(__BMI2__) || defined(__POPCNT__) || defined(__AES__) || defined(__PCLMUL__) ||        \
     defined(__LZCNT__) || defined(__XOP__) || defined(__MOVBE__))
#error "similarity build exceeds the x86-64 instruction baseline"
#endif
