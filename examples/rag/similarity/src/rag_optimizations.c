/* MySQL cosine UDF; shared float32 little-endian kernel.
 * Copyright (c) 2024
 * License: MIT
 */
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "similarity.h"
#ifndef COSINE_STANDALONE
#include <mysql.h>
#include <stdbool.h>
#endif
#ifndef COSINE_STANDALONE
/* MySQL 8 uses bool; MariaDB and older MySQL headers expose my_bool. */
#if defined(MARIADB_BASE_VERSION) || (defined(MYSQL_VERSION_ID) && MYSQL_VERSION_ID < 80000)
typedef my_bool cosine_init_result;
typedef char cosine_flag_type;
#else
typedef bool cosine_init_result;
typedef unsigned char cosine_flag_type;
#endif
cosine_init_result cosine_similarity_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (!initid || !args || args->arg_count != 2 || !args->arg_type ||
        args->arg_type[0] != STRING_RESULT || args->arg_type[1] != STRING_RESULT) {
        strcpy(message, "cosine_similarity requires two float32 little-endian blobs");
        return 1;
    }
    initid->maybe_null = 1;
    /* MySQL uses 31 to indicate an unspecified number of decimal places. */
    initid->decimals = 31;
    return 0;
}

double cosine_similarity(UDF_INIT *initid, UDF_ARGS *args, cosine_flag_type *is_null,
                         cosine_flag_type *error) {
    (void)initid;
    *is_null = 0;
    if (!args || args->arg_count != 2 || !args->args || !args->lengths) {
        *error = 1;
        return 0.0;
    }
    if (!args->args[0] || !args->args[1]) {
        *is_null = 1;
        return 0.0;
    }
#if ULONG_MAX > SIZE_MAX
    if (args->lengths[0] > SIZE_MAX || args->lengths[1] > SIZE_MAX) {
        *is_null = 1;
        return 0.0;
    }
#endif
    double result;
    if (!similarity_cosine_f32le((const unsigned char *)args->args[0],
                                 (const unsigned char *)args->args[1], (size_t)args->lengths[0],
                                 (size_t)args->lengths[1], &result)) {
        *is_null = 1;
        return 0.0;
    }
    return result;
}
void cosine_similarity_deinit(UDF_INIT *initid) { (void)initid; }
#endif
