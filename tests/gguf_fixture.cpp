/*
 * gguf_fixture.cpp -- Small deterministic GGUF metadata fixture generator.
 *
 * Purpose: exercise gguf.h readers without model weights or inference.
 * Public interface: executable path argument, which is written as a GGUF file.
 * Requirements: the pinned ggml public GGUF API and the project's native build.
 * The project license applies.
 */

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: gguf-fixture output.gguf\n");
        return 2;
    }

    gguf_context *ctx = gguf_init_empty();
    if (!ctx) return 1;

    gguf_set_val_u8(ctx, "test.scalar.u8", UINT8_MAX);
    gguf_set_val_i8(ctx, "test.scalar.i8", INT8_MIN);
    gguf_set_val_u16(ctx, "test.scalar.u16", UINT16_MAX);
    gguf_set_val_i16(ctx, "test.scalar.i16", INT16_MIN);
    gguf_set_val_u32(ctx, "test.scalar.u32", UINT32_MAX);
    gguf_set_val_i32(ctx, "test.scalar.i32", INT32_MIN);
    gguf_set_val_f32(ctx, "test.scalar.f32", 1.25f);
    gguf_set_val_bool(ctx, "test.scalar.bool", true);
    gguf_set_val_str(ctx, "test.scalar.string", "GGUF café");
    gguf_set_val_u64(ctx, "test.scalar.u64", UINT64_MAX);
    gguf_set_val_i64(ctx, "test.scalar.i64", INT64_MIN);
    gguf_set_val_f64(ctx, "test.scalar.f64", 1.0 / 3.0);

    const uint8_t u8[] = {0, 1, UINT8_MAX};
    const int8_t i8[] = {-1, 0, 1};
    const uint16_t u16[] = {0, UINT16_MAX};
    const int16_t i16[] = {INT16_MIN, 0, INT16_MAX};
    const uint32_t u32[] = {0, UINT32_MAX};
    const int32_t i32[] = {INT32_MIN, INT32_MAX};
    const float f32[] = {-0.5f, 0.0f, 3.25f};
    const int8_t booleans[] = {0, 1, 0};
    const uint64_t u64[] = {UINT64_C(9223372036854775808), UINT64_MAX};
    const int64_t i64[] = {INT64_MIN, INT64_MAX};
    const double f64[] = {-0.125, 1.0 / 3.0};
    const char *strings[] = {"uno", "dos", "tres"};

    gguf_set_arr_data(ctx, "test.array.u8", GGUF_TYPE_UINT8, u8, 3);
    gguf_set_arr_data(ctx, "test.array.i8", GGUF_TYPE_INT8, i8, 3);
    gguf_set_arr_data(ctx, "test.array.u16", GGUF_TYPE_UINT16, u16, 2);
    gguf_set_arr_data(ctx, "test.array.i16", GGUF_TYPE_INT16, i16, 3);
    gguf_set_arr_data(ctx, "test.array.u32", GGUF_TYPE_UINT32, u32, 2);
    gguf_set_arr_data(ctx, "test.array.i32", GGUF_TYPE_INT32, i32, 2);
    gguf_set_arr_data(ctx, "test.array.f32", GGUF_TYPE_FLOAT32, f32, 3);
    gguf_set_arr_data(ctx, "test.array.bool", GGUF_TYPE_BOOL, booleans, 3);
    gguf_set_arr_str(ctx, "test.array.string", strings, 3);
    gguf_set_arr_data(ctx, "test.array.u64", GGUF_TYPE_UINT64, u64, 2);
    gguf_set_arr_data(ctx, "test.array.i64", GGUF_TYPE_INT64, i64, 2);
    gguf_set_arr_data(ctx, "test.array.f64", GGUF_TYPE_FLOAT64, f64, 2);
    gguf_set_arr_data(ctx, "test.array.empty", GGUF_TYPE_INT16, NULL, 0);

    const bool written = gguf_write_to_file(ctx, argv[1], false);
    gguf_free(ctx);
    return written ? 0 : 1;
}
