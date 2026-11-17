/* Synthetic tensor-free GGUF adapter fixture for ownership tests only.
 * This is not a trained LoRA and cannot validate weight adaptation. */
#include "gguf.h"
#include <cstdio>
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    gguf_context *ctx=gguf_init_empty();
    gguf_set_val_str(ctx,"general.type","adapter");
    gguf_set_val_str(ctx,"general.architecture",argv[1]);
    gguf_set_val_str(ctx,"adapter.type","lora");
    gguf_set_val_f32(ctx,"adapter.lora.alpha",1.0f);
    bool ok=gguf_write_to_file(ctx,argv[2],false);
    gguf_free(ctx);
    return ok ? 0 : 1;
}
