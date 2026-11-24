/* Synthetic C test types only: not a MySQL or MariaDB ABI header. */
#ifndef TEST_MYSQL_UDF_H
#define TEST_MYSQL_UDF_H
#define MYSQL_VERSION_ID 80400
#include <stdbool.h>
enum Item_result { STRING_RESULT, INT_RESULT };
typedef struct { unsigned int arg_count; enum Item_result *arg_type; char **args; unsigned long *lengths; } UDF_ARGS;
typedef struct { bool maybe_null; unsigned int decimals; } UDF_INIT;
#endif
