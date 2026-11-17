/* Native synthetic control-vector probe using a real GGUF: compare logits.
 * This characterizes the upstream API independently of Tcl text generation. */
#include "llama.h"
#include <vector>
#include <cmath>
#include <cstdio>
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model *m=llama_model_load_from_file(argv[1],mp);
    if(!m) return 3;
    auto cp=llama_context_default_params();
    cp.n_ctx=512; cp.n_batch=16; cp.n_ubatch=16; cp.n_threads=2; cp.n_threads_batch=2;
    llama_context *a=llama_init_from_model(m,cp),*b=llama_init_from_model(m,cp);
    if(!a||!b) return 4;
    int dim=llama_model_n_embd(m),layers=llama_model_n_layer(m);
    std::vector<float> vector((size_t)dim*(layers-1),0.0f);vector[0]=1.0f;
    int rc=llama_set_adapter_cvec(b,vector.data(),vector.size(),dim,1,1);
    llama_token token=1;
    auto batch=llama_batch_get_one(&token,1);
    if(rc || llama_decode(a,batch) || llama_decode(b,batch)) return 5;
    const float *x=llama_get_logits_ith(a,-1),*y=llama_get_logits_ith(b,-1);
    double difference=0;
    int n=llama_vocab_n_tokens(llama_model_get_vocab(m));
    for(int i=0;i<n;++i) difference+=std::fabs((double)x[i]-y[i]);
    if(!std::isfinite(difference)||difference==0) return 6;
    llama_memory_clear(llama_get_memory(b),true);
    if(llama_set_adapter_cvec(b,NULL,0,dim,1,1) || llama_decode(b,batch)) return 7;
    y=llama_get_logits_ith(b,-1);
    double restored=0;
    for(int i=0;i<n;++i) restored+=std::fabs((double)x[i]-y[i]);
    std::printf("nonzero vector logits L1=%g; removed vector L1=%g\n",difference,restored);
    llama_free(a);llama_free(b);llama_model_free(m);llama_backend_free();
    return restored < 0.001 ? 0 : 8;
}
