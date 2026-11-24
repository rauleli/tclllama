/* SQLite loadable adapter for the shared float32 little-endian kernel. */
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1
#include "similarity.h"

static void cosine_sqlite(sqlite3_context *context, int argc, sqlite3_value **argv) {
    if (argc != 2 || sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
        sqlite3_value_type(argv[1]) != SQLITE_BLOB) {
        sqlite3_result_null(context);
        return;
    }
    int na = sqlite3_value_bytes(argv[0]);
    int nb = sqlite3_value_bytes(argv[1]);
    const unsigned char *a = sqlite3_value_blob(argv[0]);
    const unsigned char *b = sqlite3_value_blob(argv[1]);
    double value;
    if (na < 0 || nb < 0 || !similarity_cosine_f32le(a, b, (size_t)na, (size_t)nb, &value)) {
        sqlite3_result_null(context);
        return;
    }
    sqlite3_result_double(context, value);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_similarity_init(sqlite3 *db, char **error, const sqlite3_api_routines *api) {
    (void)error;
    SQLITE_EXTENSION_INIT2(api);
    return sqlite3_create_function_v2(db, "cosine_similarity_v2", 2,
                                      SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS, NULL,
                                      cosine_sqlite, NULL, NULL, NULL);
}
