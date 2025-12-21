/*
 * tclllama.c - Ik'nal v7.5 (Universal Edition - Método Ollama)
 * Detección universal de tokens de control usando atributos del vocabulario
 * Compatible con Gemma, Llama3, Mistral, Qwen, DeepSeek y cualquier modelo GGUF
 */

#include <tcl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <chrono>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------- ESTRUCTURA DE ESTADO DE IK'NAL ----------------- */
typedef struct {
    struct llama_model * model;
    struct llama_context * ctx;
    struct llama_sampler * sampler;
    const struct llama_vocab * vocab;
    
    float   temp;
    int32_t top_k;
    float   top_p;
    float   min_p;
    float   repeat_penalty;
    int32_t repeat_last_n;
    float   presence_penalty;
    float   frequency_penalty;
    int32_t mirostat;
    float   mirostat_tau;
    float   mirostat_eta;
    int32_t seed;
    
    int32_t n_predict;
    int32_t n_ctx;
    int     n_past;
    int     verbose;

    // Métricas de Telemetría (v7.0)
    double  t_eval_ms;    // Tiempo de ingestión del prompt
    double  t_gen_ms;     // Tiempo de generación de tokens
    int     n_eval;       // Tokens ingeridos (prompt)
    int     n_gen;        // Tokens generados (respuesta)
} LlamaState;

/* ----------------- VALORES POR DEFECTO ----------------- */
static void set_defaults(LlamaState *state) {
    state->temp              = 0.80f;
    state->top_k             = 40;
    state->top_p             = 0.95f;
    state->min_p             = 0.05f;
    state->repeat_penalty    = 1.10f;
    state->repeat_last_n     = 64;
    state->presence_penalty  = 0.00f;
    state->frequency_penalty = 0.00f;
    state->mirostat          = 0;
    state->mirostat_tau      = 5.00f;
    state->mirostat_eta      = 0.10f;
    state->n_predict         = -1;
    state->seed              = -1;
    state->n_past            = 0;
    state->n_ctx             = 4096;
    state->verbose           = 0;
    
    // Telemetría
    state->t_eval_ms = 0.0;
    state->t_gen_ms  = 0.0;
    state->n_eval    = 0;
    state->n_gen     = 0;
}

/* ----------------- MODULADOR DE OPCIONES (APPLY_OPTIONS) ----------------- */
static void apply_options(Tcl_Interp *interp, Tcl_Obj *options_obj, LlamaState *state) {
    Tcl_Obj *val = NULL, *k = NULL;
    #define GET_D_FLOAT(name, target) \
        k = Tcl_NewStringObj(name, -1); Tcl_IncrRefCount(k); \
        if (options_obj && Tcl_DictObjGet(interp, options_obj, k, &val) == TCL_OK && val) { \
            double d; if (Tcl_GetDoubleFromObj(interp, val, &d) == TCL_OK) target = (float)d; \
        } Tcl_DecrRefCount(k);
    #define GET_D_INT(name, target) \
        k = Tcl_NewStringObj(name, -1); Tcl_IncrRefCount(k); \
        if (options_obj && Tcl_DictObjGet(interp, options_obj, k, &val) == TCL_OK && val) { \
            int i; if (Tcl_GetIntFromObj(interp, val, &i) == TCL_OK) target = (int32_t)i; \
        } Tcl_DecrRefCount(k);

    if (options_obj) {
        GET_D_FLOAT("temperature",      state->temp);
        GET_D_INT(  "top_k",            state->top_k);
        GET_D_FLOAT("top_p",            state->top_p);
        GET_D_FLOAT("min_p",            state->min_p);
        GET_D_FLOAT("repeat_penalty",   state->repeat_penalty);
        GET_D_INT(  "repeat_last_n",    state->repeat_last_n);
        GET_D_INT(  "num_predict",      state->n_predict);
        GET_D_INT(  "mirostat",         state->mirostat);
        GET_D_FLOAT("mirostat_tau",     state->mirostat_tau);
        GET_D_FLOAT("mirostat_eta",     state->mirostat_eta);
        GET_D_INT(  "seed",             state->seed);
        
        // Validación de rangos con clamping (v6.9)
        if (state->temp < 0.0f) state->temp = 0.0f;
        if (state->temp > 2.0f) state->temp = 2.0f;
        
        if (state->top_k < 1) state->top_k = 1;
        
        if (state->top_p < 0.0f) state->top_p = 0.0f;
        if (state->top_p > 1.0f) state->top_p = 1.0f;
        
        if (state->min_p < 0.0f) state->min_p = 0.0f;
        if (state->min_p > 1.0f) state->min_p = 1.0f;
        
        if (state->repeat_penalty < 0.0f) state->repeat_penalty = 1.0f;
        
        if (state->n_predict < -1) state->n_predict = -1;
    }

    if (state->sampler != NULL) llama_sampler_free(state->sampler);
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    state->sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(state->sampler, llama_sampler_init_temp(state->temp));
    llama_sampler_chain_add(state->sampler, llama_sampler_init_top_k(state->top_k));
    llama_sampler_chain_add(state->sampler, llama_sampler_init_top_p(state->top_p, 1));
    llama_sampler_chain_add(state->sampler, llama_sampler_init_min_p(state->min_p, 1));
    llama_sampler_chain_add(state->sampler, llama_sampler_init_penalties(state->repeat_last_n, state->repeat_penalty, state->presence_penalty, state->frequency_penalty));
    if (state->mirostat == 1) llama_sampler_chain_add(state->sampler, llama_sampler_init_mirostat(llama_vocab_n_tokens(state->vocab), state->seed, state->mirostat_tau, state->mirostat_eta, 100));
    else if (state->mirostat == 2) llama_sampler_chain_add(state->sampler, llama_sampler_init_mirostat_v2(state->seed, state->mirostat_tau, state->mirostat_eta));
    llama_sampler_chain_add(state->sampler, llama_sampler_init_dist(state->seed));
}

static void fill_batch(struct llama_batch & batch, llama_token id, int pos, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens]   = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

/* ----------------- UTF-8 VALIDATION HELPERS ----------------- */
static int utf8_char_length(unsigned char first_byte) {
    // Determinar cuántos bytes tiene un carácter UTF-8 basado en el primer byte
    if ((first_byte & 0x80) == 0x00) return 1;      // 0xxxxxxx - ASCII
    if ((first_byte & 0xE0) == 0xC0) return 2;      // 110xxxxx - 2 bytes
    if ((first_byte & 0xF0) == 0xE0) return 3;      // 1110xxxx - 3 bytes
    if ((first_byte & 0xF8) == 0xF0) return 4;      // 11110xxx - 4 bytes
    return 1; // Inválido, tratar como 1 byte
}

static size_t find_last_utf8_boundary(const std::string& str, size_t max_pos) {
    // Encontrar la posición del último carácter UTF-8 completo antes de max_pos
    if (max_pos >= str.length()) {
        max_pos = str.length();
    }
    
    size_t pos = 0;
    size_t last_valid = 0;
    
    while (pos < max_pos) {
        int char_len = utf8_char_length((unsigned char)str[pos]);
        
        if (pos + char_len <= max_pos) {
            // Este carácter cabe completo
            last_valid = pos + char_len;
            pos += char_len;
        } else {
            // Este carácter se cortaría, detener
            break;
        }
    }
    
    return last_valid;
}

/* ----------------- CORE GENERATION LOOP (v7.5 - Universal + Buffer) ----------------- */
static int run_inference(Tcl_Interp *interp, LlamaState *state, const char *cb_name, 
                        std::vector<llama_token> & stop_ids) {
    Tcl_DString resp;
    Tcl_DStringInit(&resp);
    
    // Buffer para detectar tags textuales (caso Gemma)
    std::string text_buffer;
    const size_t BUFFER_SIZE = 50;
    
    auto t_start_gen = std::chrono::high_resolution_clock::now();
    
    int max_tokens = (state->n_predict > 0) ? state->n_predict : 4096;
    int p_cnt = 0;
    
    while (p_cnt < max_tokens) {
        if (state->n_past >= state->n_ctx) break;
        
        llama_token id = llama_sampler_sample(state->sampler, state->ctx, -1);
        llama_sampler_accept(state->sampler, id);
        
        // DEBUG: Si verbose está activado, mostrar info del token
        if (state->verbose) {
            enum llama_token_attr attr = llama_vocab_get_attr(state->vocab, id);
            char debug_piece[64];
            int debug_n = llama_token_to_piece(state->vocab, id, debug_piece, 63, 0, false);
            if (debug_n > 0) debug_piece[debug_n] = 0;
            else debug_piece[0] = 0;
            
            fprintf(stderr, "[Ik'nal DEBUG] token=%d, attr=%d, piece='%s', is_eog=%d, is_control=%d\n",
                    id, attr, debug_piece, 
                    llama_vocab_is_eog(state->vocab, id),
                    (attr & LLAMA_TOKEN_ATTR_CONTROL) ? 1 : 0);
        }
        
        // --- ESCUDO NIVEL 1: TOKENS OFICIALES DE CONTROL ---
        
        // 1a. EOG nativo (funciona para la mayoría)
        if (llama_vocab_is_eog(state->vocab, id)) break;
        
        // 1b. Token marcado como CONTROL (Llama3, Mistral, Qwen bien configurados)
        if (llama_vocab_get_attr(state->vocab, id) & LLAMA_TOKEN_ATTR_CONTROL) break;
        
        // --- ESCUDO NIVEL 2: STOP IDS MANUALES ---
        bool s_stop = false;
        for (auto s : stop_ids) {
            if (id == s) {
                s_stop = true;
                break;
            }
        }
        if (s_stop) break;
        
        char piece[512];
        int n = llama_token_to_piece(state->vocab, id, piece, sizeof(piece) - 1, 0, false);
        
        if (n > 0 && n < (int)sizeof(piece)) {
            piece[n] = 0;
            
            // Agregar al buffer de texto
            text_buffer.append(piece, n);
            
            // Mantener buffer limitado
            if (text_buffer.length() > BUFFER_SIZE) {
                text_buffer = text_buffer.substr(text_buffer.length() - BUFFER_SIZE);
            }
            
            // --- ESCUDO NIVEL 3: DETECCIÓN TEXTUAL (caso Gemma) ---
            // Buscar tags generados como texto fragmentado
            bool found_tag = false;
            
            if (text_buffer.find("<end_of_turn>") != std::string::npos ||
                text_buffer.find("<start_of_turn>") != std::string::npos ||
                text_buffer.find("<|im_end|>") != std::string::npos ||
                text_buffer.find("<|eot_id|>") != std::string::npos ||
                text_buffer.find("<|endoftext|>") != std::string::npos) {
                found_tag = true;
            }
            
            if (found_tag) {
                // Encontramos un tag en el texto
                // Necesitamos remover el tag del output
                
                // Calcular cuánto del buffer es el tag
                size_t tag_pos = std::string::npos;
                std::string found_tag_str;
                
                const char* tags[] = {
                    "<end_of_turn>", "<start_of_turn>", 
                    "<|im_end|>", "<|eot_id|>", "<|endoftext|>", NULL
                };
                
                for (int i = 0; tags[i] != NULL; i++) {
                    size_t pos = text_buffer.find(tags[i]);
                    if (pos != std::string::npos) {
                        tag_pos = pos;
                        found_tag_str = tags[i];
                        break;
                    }
                }
                
                if (tag_pos != std::string::npos) {
                    // Extraer solo lo que va antes del tag
                    std::string before_tag = text_buffer.substr(0, tag_pos);
                    
                    if (!before_tag.empty()) {
                        // Asegurar UTF-8 válido
                        size_t safe_len = find_last_utf8_boundary(before_tag, before_tag.length());
                        if (safe_len > 0) {
                            Tcl_DStringAppend(&resp, before_tag.c_str(), safe_len);
                            
                            if (cb_name) {
                                Tcl_Obj *cmd = Tcl_NewListObj(0, NULL);
                                Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(cb_name, -1));
                                Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(before_tag.c_str(), safe_len));
                                Tcl_EvalObjEx(interp, cmd, TCL_EVAL_DIRECT);
                            }
                        }
                    }
                }
                break;
            }
            
            // Enviar caracteres de forma segura (UTF-8)
            if (text_buffer.length() > 20) {
                size_t safe_len = text_buffer.length() - 20;
                safe_len = find_last_utf8_boundary(text_buffer, safe_len);
                
                if (safe_len > 0) {
                    std::string safe_part = text_buffer.substr(0, safe_len);
                    
                    Tcl_DStringAppend(&resp, safe_part.c_str(), safe_part.length());
                    
                    if (cb_name) {
                        Tcl_Obj *cmd = Tcl_NewListObj(0, NULL);
                        Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(cb_name, -1));
                        Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(safe_part.c_str(), -1));
                        
                        if (Tcl_EvalObjEx(interp, cmd, TCL_EVAL_DIRECT) != TCL_OK) {
                            Tcl_DStringFree(&resp);
                            return TCL_ERROR;
                        }
                    }
                    
                    text_buffer = text_buffer.substr(safe_len);
                }
            }
        }
        
        struct llama_batch b = llama_batch_init(1, 0, 1);
        fill_batch(b, id, state->n_past, true);
        state->n_past++;
        
        if (llama_decode(state->ctx, b) != 0) {
            llama_batch_free(b);
            Tcl_DStringFree(&resp);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed during generation", -1));
            return TCL_ERROR;
        }
        llama_batch_free(b);
        p_cnt++;
    }
    
    // Al terminar, enviar lo que quedó en el buffer (sin tags)
    if (!text_buffer.empty()) {
        // Verificar que no haya tags parciales al final
        bool has_partial_tag = false;
        const char* partial_tags[] = {"<end", "<start", "<|im", "<|eot", NULL};
        
        for (int i = 0; partial_tags[i] != NULL; i++) {
            if (text_buffer.find(partial_tags[i]) != std::string::npos) {
                has_partial_tag = true;
                break;
            }
        }
        
        if (!has_partial_tag) {
            Tcl_DStringAppend(&resp, text_buffer.c_str(), text_buffer.length());
            
            if (cb_name) {
                Tcl_Obj *cmd = Tcl_NewListObj(0, NULL);
                Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(cb_name, -1));
                Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(text_buffer.c_str(), -1));
                Tcl_EvalObjEx(interp, cmd, TCL_EVAL_DIRECT);
            }
        }
    }
    
    auto t_end_gen = std::chrono::high_resolution_clock::now();
    state->t_gen_ms = std::chrono::duration<double, std::milli>(t_end_gen - t_start_gen).count();
    state->n_gen = p_cnt;
    
    Tcl_SetObjResult(interp, Tcl_NewStringObj(Tcl_DStringValue(&resp), -1));
    Tcl_DStringFree(&resp);
    return TCL_OK;
}

/* ----------------- LLAMA::GENERATE (Stateful) ----------------- */
static int Llama_Generate_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::generate handle prompt ?-callback proc? ?-options dict? ?-reset bool? ?-stop_ids list? ?-system string? ?-max_tokens int?", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;

    const char *prompt = Tcl_GetString(objv[2]);
    char *cb_name = NULL;
    char *system_msg = NULL;
    int reset = 0;
    std::vector<llama_token> stop_ids;

    for (int i = 3; i < objc; i += 2) {
        if (i + 1 >= objc) break;
        const char *opt = Tcl_GetString(objv[i]);
        if (strcmp(opt, "-callback") == 0) cb_name = Tcl_GetString(objv[i+1]);
        if (strcmp(opt, "-options") == 0) apply_options(interp, objv[i+1], state);
        if (strcmp(opt, "-reset") == 0) Tcl_GetBooleanFromObj(interp, objv[i+1], &reset);
        if (strcmp(opt, "-system") == 0) system_msg = Tcl_GetString(objv[i+1]);
        if (strcmp(opt, "-max_tokens") == 0) {
            int max_tokens;
            if (Tcl_GetIntFromObj(interp, objv[i+1], &max_tokens) == TCL_OK) {
                state->n_predict = max_tokens;
            }
        }
        if (strcmp(opt, "-stop_ids") == 0) {
            int se; Tcl_Obj **sel;
            if (Tcl_ListObjGetElements(interp, objv[i+1], &se, &sel) == TCL_OK) {
                for (int k=0; k<se; k++) {
                    int id;
                    if (Tcl_GetIntFromObj(interp, sel[k], &id) == TCL_OK) {
                        stop_ids.push_back(id);
                    }
                }
            }
        }
    }

    if (reset) {
        state->n_past = 0;
        llama_kv_self_clear(state->ctx);
    }
    
    // Construir prompt con system message si existe
    std::string full_prompt;
    if (system_msg && strlen(system_msg) > 0 && state->n_past == 0) {
        full_prompt = std::string(system_msg) + "\n\n" + std::string(prompt);
    } else {
        full_prompt = std::string(prompt);
    }
    
    std::vector<llama_token> tokens(full_prompt.length() + 256);
    int n_tok = llama_tokenize(state->vocab, full_prompt.c_str(), full_prompt.length(), 
                                tokens.data(), tokens.size(), (state->n_past == 0), false);
    
    if (n_tok < 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Tokenization failed", -1));
        return TCL_ERROR;
    }

    // Verificar overflow de contexto
    if (state->n_past + n_tok >= state->n_ctx) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Context overflow: n_past=%d + n_tok=%d >= n_ctx=%d", 
                 state->n_past, n_tok, state->n_ctx);
        Tcl_SetObjResult(interp, Tcl_NewStringObj(msg, -1));
        return TCL_ERROR;
    }

    // Telemetría: medir tiempo de ingestión del prompt
    auto t_start_eval = std::chrono::high_resolution_clock::now();
    
    struct llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; i++) {
        fill_batch(batch, tokens[i], state->n_past, (i == n_tok - 1));
        state->n_past++;
    }
    
    if (llama_decode(state->ctx, batch) != 0) {
        llama_batch_free(batch);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed", -1));
        return TCL_ERROR;
    }
    llama_batch_free(batch);
    
    auto t_end_eval = std::chrono::high_resolution_clock::now();
    state->t_eval_ms = std::chrono::duration<double, std::milli>(t_end_eval - t_start_eval).count();
    state->n_eval = n_tok;

    return run_inference(interp, state, cb_name, stop_ids);
}

/* ----------------- LLAMA::CHAT (Stateless) ----------------- */
static int Llama_Chat(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::chat handle messages ?-callback proc? ?-options dict? ?-stop_ids list? ?-max_tokens int?", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;

    // CHAT ES STATELESS: RESET SIEMPRE
    state->n_past = 0;
    llama_kv_self_clear(state->ctx);

    int n_msgs;
    Tcl_Obj **msgs_elems;
    if (Tcl_ListObjGetElements(interp, objv[2], &n_msgs, &msgs_elems) != TCL_OK) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid messages list", -1));
        return TCL_ERROR;
    }
    
    std::vector<llama_chat_message> cmsgs;
    std::vector<char*> allocated_strings;
    
    for (int i = 0; i < n_msgs; i++) {
        Tcl_Obj *v_role, *v_content;
        Tcl_Obj *role_key = Tcl_NewStringObj("role", -1);
        Tcl_Obj *content_key = Tcl_NewStringObj("content", -1);
        
        Tcl_IncrRefCount(role_key);
        Tcl_IncrRefCount(content_key);
        
        Tcl_DictObjGet(interp, msgs_elems[i], role_key, &v_role);
        Tcl_DictObjGet(interp, msgs_elems[i], content_key, &v_content);
        
        Tcl_DecrRefCount(role_key);
        Tcl_DecrRefCount(content_key);
        
        if (v_role && v_content) {
            char* role_str = strdup(Tcl_GetString(v_role));
            char* content_str = strdup(Tcl_GetString(v_content));
            
            if (!role_str || !content_str) {
                // Cleanup on allocation failure
                for (auto ptr : allocated_strings) free(ptr);
                free(role_str);
                free(content_str);
                Tcl_SetObjResult(interp, Tcl_NewStringObj("Memory allocation failed", -1));
                return TCL_ERROR;
            }
            
            allocated_strings.push_back(role_str);
            allocated_strings.push_back(content_str);
            cmsgs.push_back({role_str, content_str});
        }
    }

    char *cb_name = NULL;
    std::vector<llama_token> stop_ids;
    
    for (int i = 3; i < objc; i += 2) {
        if (i + 1 >= objc) break;
        const char *opt = Tcl_GetString(objv[i]);
        if (strcmp(opt, "-callback") == 0) cb_name = Tcl_GetString(objv[i+1]);
        if (strcmp(opt, "-options") == 0) apply_options(interp, objv[i+1], state);
        if (strcmp(opt, "-max_tokens") == 0) {
            int max_tokens;
            if (Tcl_GetIntFromObj(interp, objv[i+1], &max_tokens) == TCL_OK) {
                state->n_predict = max_tokens;
            }
        }
        if (strcmp(opt, "-stop_ids") == 0) {
            int se; Tcl_Obj **sel;
            if (Tcl_ListObjGetElements(interp, objv[i+1], &se, &sel) == TCL_OK) {
                for (int k=0; k<se; k++) {
                    int id;
                    if (Tcl_GetIntFromObj(interp, sel[k], &id) == TCL_OK) {
                        stop_ids.push_back(id);
                    }
                }
            }
        }
    }

    // Obtener template del modelo con buffer dinámico
    char tmpl_buffer[256];
    int tmpl_ret = llama_model_meta_val_str(state->model, "tokenizer.chat_template", 
                                             tmpl_buffer, sizeof(tmpl_buffer));
    
    std::string tmpl;
    if (tmpl_ret > 0 && tmpl_ret < (int)sizeof(tmpl_buffer)) {
        tmpl = std::string(tmpl_buffer);
    } else if (tmpl_ret > (int)sizeof(tmpl_buffer)) {
        // Template es más grande, usar buffer dinámico
        std::vector<char> large_buffer(tmpl_ret + 1);
        llama_model_meta_val_str(state->model, "tokenizer.chat_template", 
                                  large_buffer.data(), large_buffer.size());
        tmpl = std::string(large_buffer.data());
    } else {
        // Sin template, usar fallback simple
        tmpl = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}";
    }

    int32_t fmt_len = llama_chat_apply_template(tmpl.c_str(), cmsgs.data(), cmsgs.size(), 
                                                  true, NULL, 0);
    
    if (fmt_len < 0) {
        for (auto ptr : allocated_strings) free(ptr);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Template application failed", -1));
        return TCL_ERROR;
    }
    
    std::vector<char> formatted(fmt_len + 1);
    llama_chat_apply_template(tmpl.c_str(), cmsgs.data(), cmsgs.size(), true, 
                               formatted.data(), fmt_len + 1);
    
    // Liberar strings asignados
    for (auto ptr : allocated_strings) free(ptr);

    std::vector<llama_token> tokens(fmt_len + 1024);
    int n_tok = llama_tokenize(state->vocab, formatted.data(), fmt_len, 
                                tokens.data(), tokens.size(), true, false);
    
    if (n_tok < 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Tokenization failed", -1));
        return TCL_ERROR;
    }

    // Telemetría: medir tiempo de ingestión
    auto t_start_eval = std::chrono::high_resolution_clock::now();
    
    struct llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; i++) {
        fill_batch(batch, tokens[i], state->n_past, (i == n_tok - 1));
        state->n_past++;
    }
    
    if (llama_decode(state->ctx, batch) != 0) {
        llama_batch_free(batch);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed", -1));
        return TCL_ERROR;
    }
    llama_batch_free(batch);
    
    auto t_end_eval = std::chrono::high_resolution_clock::now();
    state->t_eval_ms = std::chrono::duration<double, std::milli>(t_end_eval - t_start_eval).count();
    state->n_eval = n_tok;

    return run_inference(interp, state, cb_name, stop_ids);
}

/* ----------------- LLAMA::DETOKENIZE (v7.0) ----------------- */
static int Llama_Detokenize_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::detokenize handle token_list", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    int n_tokens;
    Tcl_Obj **token_elems;
    if (Tcl_ListObjGetElements(interp, objv[2], &n_tokens, &token_elems) != TCL_OK) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid token list", -1));
        return TCL_ERROR;
    }
    
    Tcl_DString result;
    Tcl_DStringInit(&result);
    
    for (int i = 0; i < n_tokens; i++) {
        int token_id;
        if (Tcl_GetIntFromObj(interp, token_elems[i], &token_id) != TCL_OK) {
            Tcl_DStringFree(&result);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid token ID", -1));
            return TCL_ERROR;
        }
        
        char piece[256];
        int n = llama_token_to_piece(state->vocab, token_id, piece, sizeof(piece) - 1, 0, false);
        if (n > 0) {
            piece[n] = 0;
            Tcl_DStringAppend(&result, piece, n);
        }
    }
    
    Tcl_SetObjResult(interp, Tcl_NewStringObj(Tcl_DStringValue(&result), -1));
    Tcl_DStringFree(&result);
    return TCL_OK;
}

/* ----------------- LLAMA::VERSION - Información de versión ----------------- */
static int Llama_Version_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 1) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::version", -1));
        return TCL_ERROR;
    }
    
    Tcl_Obj *dict = Tcl_NewDictObj();
    
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("version", -1),
                   Tcl_NewStringObj("7.5", -1));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("edition", -1),
                   Tcl_NewStringObj("AeroAlebrije Universal Edition", -1));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("compiled", -1),
                   Tcl_NewStringObj(__DATE__ " " __TIME__, -1));
    
    Tcl_SetObjResult(interp, dict);
    return TCL_OK;
}

/* ----------------- LLAMA::INFO - Información completa (v6.9 + v7.0) ----------------- */
static int Llama_Info_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::info handle", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    Tcl_Obj *dict = Tcl_NewDictObj();
    
    // Información del contexto (v6.9)
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx", -1), 
                   Tcl_NewIntObj(llama_n_ctx(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_past", -1),
                   Tcl_NewIntObj(state->n_past));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx_used", -1),
                   Tcl_NewIntObj(state->n_past));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx_available", -1),
                   Tcl_NewIntObj(state->n_ctx - state->n_past));
    
    // Información del modelo (v6.9)
    char model_desc[256];
    llama_model_desc(state->model, model_desc, sizeof(model_desc));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("model_desc", -1),
                   Tcl_NewStringObj(model_desc, -1));
    
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_vocab", -1),
                   Tcl_NewIntObj(llama_vocab_n_tokens(state->vocab)));
    
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("model_size", -1),
                   Tcl_NewWideIntObj(llama_model_size(state->model)));
    
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("model_n_params", -1),
                   Tcl_NewWideIntObj(llama_model_n_params(state->model)));
    
    // Parámetros de sampling actuales (v6.9)
    Tcl_Obj *sampling = Tcl_NewDictObj();
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("temperature", -1),
                   Tcl_NewDoubleObj(state->temp));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("top_k", -1),
                   Tcl_NewIntObj(state->top_k));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("top_p", -1),
                   Tcl_NewDoubleObj(state->top_p));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("min_p", -1),
                   Tcl_NewDoubleObj(state->min_p));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("repeat_penalty", -1),
                   Tcl_NewDoubleObj(state->repeat_penalty));
    
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("sampling", -1), sampling);
    
    // Telemetría (v7.0) - Con protección contra división por cero
    Tcl_Obj *telemetry = Tcl_NewDictObj();
    Tcl_DictObjPut(interp, telemetry, Tcl_NewStringObj("t_eval_ms", -1),
                   Tcl_NewDoubleObj(state->t_eval_ms));
    Tcl_DictObjPut(interp, telemetry, Tcl_NewStringObj("t_gen_ms", -1),
                   Tcl_NewDoubleObj(state->t_gen_ms));
    Tcl_DictObjPut(interp, telemetry, Tcl_NewStringObj("n_eval", -1),
                   Tcl_NewIntObj(state->n_eval));
    Tcl_DictObjPut(interp, telemetry, Tcl_NewStringObj("n_gen", -1),
                   Tcl_NewIntObj(state->n_gen));
    
    // TPS con protección contra división por cero
    double eval_tps = (state->t_eval_ms > 0.0) 
        ? (state->n_eval / (state->t_eval_ms / 1000.0)) 
        : 0.0;
    double gen_tps = (state->t_gen_ms > 0.0) 
        ? (state->n_gen / (state->t_gen_ms / 1000.0)) 
        : 0.0;
    
    Tcl_DictObjPut(interp, telemetry, Tcl_NewStringObj("eval_tps", -1),
                   Tcl_NewDoubleObj(eval_tps));
    Tcl_DictObjPut(interp, telemetry, Tcl_NewStringObj("gen_tps", -1),
                   Tcl_NewDoubleObj(gen_tps));
    
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("telemetry", -1), telemetry);
    
    Tcl_SetObjResult(interp, dict);
    return TCL_OK;
}

/* ----------------- LLAMA::VERBOSE - Control de verbosidad ----------------- */
static int Llama_Verbose_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 2 || objc > 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::verbose handle ?0|1?", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    if (objc == 3) {
        // Set verbose
        int verbose;
        if (Tcl_GetBooleanFromObj(interp, objv[2], &verbose) != TCL_OK) {
            return TCL_ERROR;
        }
        state->verbose = verbose;
    }
    
    // Return current verbose state
    Tcl_SetObjResult(interp, Tcl_NewIntObj(state->verbose));
    return TCL_OK;
}

/* ----------------- DIAGNÓSTICO: LLAMA::GET_CONTEXT ----------------- */
static int Llama_GetContext_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::get_context handle", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    Tcl_SetObjResult(interp, Tcl_NewIntObj(state->n_past));
    return TCL_OK;
}

/* ----------------- SOPORTE E INIT ----------------- */
static int Llama_Tokenize_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::tokenize handle text", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    const char *text = Tcl_GetString(objv[2]);
    std::vector<llama_token> tokens(strlen(text) + 256);
    int n = llama_tokenize(state->vocab, text, strlen(text), tokens.data(), tokens.size(), true, false);
    
    if (n < 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Tokenization failed", -1));
        return TCL_ERROR;
    }
    
    Tcl_Obj *res = Tcl_NewListObj(0, NULL);
    for (int i = 0; i < n; i++) {
        Tcl_ListObjAppendElement(interp, res, Tcl_NewIntObj(tokens[i]));
    }
    Tcl_SetObjResult(interp, res);
    return TCL_OK;
}

static int Llama_ClearCache_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::clear_cache handle", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    llama_kv_self_clear(state->ctx);
    state->n_past = 0;
    
    // Reset telemetría
    state->t_eval_ms = 0.0;
    state->t_gen_ms = 0.0;
    state->n_eval = 0;
    state->n_gen = 0;
    
    return TCL_OK;
}

static int Llama_Init_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 2 || objc > 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::init model_path ?n_ctx?", -1));
        return TCL_ERROR;
    }
    
    const char *model_path = Tcl_GetString(objv[1]);
    int n_ctx = 4096;
    
    if (objc == 3) {
        if (Tcl_GetIntFromObj(interp, objv[2], &n_ctx) != TCL_OK) {
            return TCL_ERROR;
        }
        if (n_ctx < 512 || n_ctx > 32768) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("n_ctx must be between 512 and 32768", -1));
            return TCL_ERROR;
        }
    }
    
    LlamaState *state = (LlamaState*)ckalloc(sizeof(LlamaState));
    if (!state) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Memory allocation failed", -1));
        return TCL_ERROR;
    }
    
    memset(state, 0, sizeof(LlamaState));
    set_defaults(state);
    state->n_ctx = n_ctx;
    
    llama_model_params mparams = llama_model_default_params();
    state->model = llama_model_load_from_file(model_path, mparams);
    
    if (!state->model) {
        ckfree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to load model", -1));
        return TCL_ERROR;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 2048;
    
    state->ctx = llama_init_from_model(state->model, cparams);
    
    if (!state->ctx) {
        llama_model_free(state->model);
        ckfree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create context", -1));
        return TCL_ERROR;
    }
    
    state->vocab = llama_model_get_vocab(state->model);
    apply_options(interp, NULL, state);
    
    char handle[64];
    snprintf(handle, sizeof(handle), "llama%p", (void*)state);
    Tcl_CreateObjCommand(interp, handle, NULL, state, NULL);
    
    Tcl_SetObjResult(interp, Tcl_NewStringObj(handle, -1));
    return TCL_OK;
}

static int Llama_Free_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::free handle", -1));
        return TCL_ERROR;
    }
    
    Tcl_CmdInfo info;
    if (Tcl_GetCommandInfo(interp, Tcl_GetString(objv[1]), &info) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }
    LlamaState *state = (LlamaState*)info.objClientData;
    
    if (state->ctx) llama_free(state->ctx);
    if (state->model) llama_model_free(state->model);
    if (state->sampler) llama_sampler_free(state->sampler);
    
    ckfree((char*)state);
    Tcl_DeleteCommand(interp, Tcl_GetString(objv[1]));
    
    return TCL_OK;
}

int Tclllama_Init(Tcl_Interp *interp) {
    if (Tcl_InitStubs(interp, "8.6", 0) == NULL) {
        return TCL_ERROR;
    }
    
    Tcl_CreateObjCommand(interp, "llama::version", Llama_Version_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::init", Llama_Init_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::free", Llama_Free_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::generate", Llama_Generate_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::chat", Llama_Chat, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::tokenize", Llama_Tokenize_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::detokenize", Llama_Detokenize_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::clear_cache", Llama_ClearCache_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::get_context", Llama_GetContext_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::info", Llama_Info_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::verbose", Llama_Verbose_Cmd, NULL, NULL);
    
    return Tcl_PkgProvide(interp, "tclllama", "7.5");
}

#ifdef __cplusplus
}
#endif
