/* Unit test with synthetic UDF types, not a server integration test. */
#include "../src/similarity.c"
#include "../src/rag_optimizations.c"
#include <stdio.h>
#include <stdlib.h>
static void check(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}
int main(void) {
    UDF_INIT init = {0};
    enum Item_result types[] = {STRING_RESULT, STRING_RESULT};
    unsigned char one[] = {0, 0, 0x80, 0x3f}, zero[] = {0, 0, 0, 0}, nan[] = {0, 0, 0xc0, 0x7f};
    char *values[] = {(char *)one, (char *)one};
    unsigned long lengths[] = {4, 4};
    UDF_ARGS args = {2, types, values, lengths};
    char message[256] = {0};
    cosine_flag_type is_null = 0, error = 0;
    check(!cosine_similarity_init(&init, &args, message) && init.maybe_null && init.decimals == 31,
          "init");
    check(cosine_similarity(&init, &args, &is_null, &error) == 1.0 && !is_null && !error,
          "valid vector");
    args.arg_count = 1;
    check(cosine_similarity_init(&init, &args, message), "argument count");
    args.arg_count = 2;
    types[1] = INT_RESULT;
    check(cosine_similarity_init(&init, &args, message), "argument type");
    types[1] = STRING_RESULT;
    values[0] = NULL;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(is_null && !error, "NULL input");
    values[0] = (char *)one;
    lengths[1] = 3;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(is_null && !error, "invalid length");
    lengths[1] = 4;
    lengths[1] = 8;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(is_null && !error, "different dimensions");
    lengths[1] = 4;
    lengths[0] = lengths[1] = 0;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(is_null && !error, "empty vectors");
    lengths[0] = lengths[1] = 4;
    values[0] = values[1] = (char *)zero;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(is_null && !error, "zero vector");
    values[0] = values[1] = (char *)nan;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(is_null && !error, "NaN vector");
    values[0] = values[1] = (char *)one;
    cosine_similarity(&init, &args, &is_null, &error);
    check(!is_null && !error, "valid row after invalid");
    args.args = NULL;
    is_null = error = 0;
    cosine_similarity(&init, &args, &is_null, &error);
    check(error && !is_null, "structural error");
    puts("synthetic UDF contract tests passed; no real ABI or server tested");
    return 0;
}
