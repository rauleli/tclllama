/*
 * tclllama.c - Tcl binding for llama.cpp, package version 7.5.
 *
 * Public commands are registered by Tclllama_Init. This file owns model
 * handles and implements model loading, generation, chat, tokenization and
 * related inspection commands. It requires Tcl 8.6 and the llama.cpp API
 * revision selected by the project build. See API.md for the public command
 * contract. The project license applies.
 */

#include <tcl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <limits>
#include <cmath>
#include <new>
#include <memory>

#include "llama.h"
#include "ggml-backend.h"
#include "stream_buffer.h"
#include "sampling_guard.h"
#include "schema_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SamplingOptions {
    float temp, top_p, min_p, repeat_penalty;
    float presence_penalty, frequency_penalty, mirostat_tau, mirostat_eta;
    float typical_p, dry_multiplier, dry_base, xtc_probability, xtc_threshold;
    int32_t top_k, repeat_last_n, n_predict, mirostat, seed;
    int32_t dry_allowed_length, dry_penalty_last_n;
    std::string grammar;
    std::string json_schema;
    std::vector<llama_logit_bias> logit_bias;
    std::vector<std::string> dry_sequence_breakers;
};

/* A sequence owns its sampler, immutable-after-start options and last logits. */
struct SequenceSampling {
    SamplingOptions options;
    llama_sampler *sampler;
    std::vector<float> logits;
    bool started, finished;
    explicit SequenceSampling(const SamplingOptions &o) : options(o), sampler(NULL), started(false), finished(false) {}
    ~SequenceSampling() { if (sampler) llama_sampler_free(sampler); }
};

/* A snapshot belongs exclusively to its handle and is never exposed as bytes. */
struct InferenceSnapshot {
    uint8_t *data;
    size_t size;
    int n_past;
    SamplingOptions options;
    struct llama_sampler *sampler;
};

static void FreeSnapshot(InferenceSnapshot *snapshot) {
    if (!snapshot) return;
    if (snapshot->sampler) llama_sampler_free(snapshot->sampler);
    free(snapshot->data);
    delete snapshot;
}

/* ----------------- ESTRUCTURA DE ESTADO DE IK'NAL ----------------- */
typedef struct {
    Tcl_Interp *interp;
    Tcl_Command command;
    int closing;
    int busy;
    struct llama_model * model;
    struct llama_context * ctx;
    struct llama_sampler * sampler;
    const struct llama_vocab * vocab;
    struct llama_context_params context_params;
    InferenceSnapshot *snapshot;
    std::vector<int> *sequence_positions; // -1: unregistered; sequence-local positions
    std::vector<SequenceSampling *> *sequence_sampling;
    bool sequence_generation;
    bool sequence_fork_pending;
    std::string *grammar;
    std::string *json_schema;
    std::vector<llama_logit_bias> *logit_bias;
    std::vector<std::string> *dry_sequence_breakers;
    
    float   temp;
    float   typical_p;
    float   dry_multiplier;
    float   dry_base;
    float   xtc_probability;
    float   xtc_threshold;
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
    int32_t dry_allowed_length;
    int32_t dry_penalty_last_n;
    
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

static std::atomic<unsigned long long> next_handle_id(1);

static void LlamaStateFree(char *data) {
    LlamaState *state = (LlamaState *)data;
    FreeSnapshot(state->snapshot);
    if (state->sequence_sampling) {
        for (SequenceSampling *sampling : *state->sequence_sampling) delete sampling;
        delete state->sequence_sampling;
    }
    delete state->sequence_positions;
    delete state->grammar;
    delete state->json_schema;
    delete state->logit_bias;
    delete state->dry_sequence_breakers;
    if (state->sampler) llama_sampler_free(state->sampler);
    if (state->ctx) llama_free(state->ctx);
    if (state->model) llama_model_free(state->model);
    ckfree(data);
}

static void LlamaHandleDelete(ClientData data) {
    LlamaState *state = (LlamaState *)data;
    state->closing = 1;
    state->command = NULL;
    Tcl_EventuallyFree((ClientData)state, LlamaStateFree);
}

static int LlamaHandleCmd(ClientData data, Tcl_Interp *interp, int objc,
                          Tcl_Obj *const objv[]) {
    (void)data; (void)objc; (void)objv;
    Tcl_SetObjResult(interp, Tcl_NewStringObj("Handle is not directly callable", -1));
    return TCL_ERROR;
}

static LlamaState *GetLlamaState(Tcl_Interp *interp, Tcl_Obj *name, bool allow_busy = false) {
    Tcl_Command command = Tcl_GetCommandFromObj(interp, name);
    Tcl_CmdInfo info;
    if (command && Tcl_GetCommandInfoFromToken(command, &info) &&
        info.isNativeObjectProc && info.objProc == LlamaHandleCmd &&
        info.deleteProc == LlamaHandleDelete && info.objClientData &&
        info.deleteData == info.objClientData) {
        LlamaState *state = (LlamaState *)info.objClientData;
        if (state->interp == interp && state->command == command && !state->closing) {
            if (state->busy && !allow_busy) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj("Handle is busy", -1));
                return NULL;
            }
            return state;
        }
    }
    Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
    return NULL;
}

class LlamaOperation {
    LlamaState *state;
public:
    explicit LlamaOperation(LlamaState *value) : state(value) {
        Tcl_Preserve((ClientData)state);
        state->busy = 1;
    }
    ~LlamaOperation() {
        state->busy = 0;
        Tcl_Release((ClientData)state);
    }
};

static int CheckHandleAfterCallback(Tcl_Interp *interp, LlamaState *state) {
    if (Tcl_InterpDeleted(interp)) return TCL_ERROR;
    if (!state->closing) return TCL_OK;
    Tcl_SetObjResult(interp, Tcl_NewStringObj("Handle was deleted during callback", -1));
    return TCL_ERROR;
}

static int InvokeCallback(Tcl_Interp *interp, LlamaState *state,
                          const char *name, const char *text, int length) {
    Tcl_Obj *cmd = Tcl_NewListObj(0, NULL);
    Tcl_IncrRefCount(cmd);
    Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(name, -1));
    Tcl_ListObjAppendElement(interp, cmd, Tcl_NewStringObj(text, length));
    int result = Tcl_EvalObjEx(interp, cmd, TCL_EVAL_DIRECT);
    Tcl_DecrRefCount(cmd);
    if (CheckHandleAfterCallback(interp, state) != TCL_OK) return TCL_ERROR;
    return result;
}

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
    state->typical_p         = 1.00f;
    state->dry_multiplier    = 0.00f;
    state->dry_base          = 1.75f;
    state->dry_allowed_length = 2;
    state->dry_penalty_last_n = 64;
    state->xtc_probability   = 0.00f;
    state->xtc_threshold     = 0.10f;
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

/* ----------------- SAMPLING OPTION PARSING -----------------
 * These are the sampling keys accepted by the current public dict syntax.
 * Parse into a copy first so a malformed dict cannot partly change a handle.
 */
static SamplingOptions SamplingOptionsFromState(const LlamaState *state) {
    SamplingOptions options;
    options.temp = state->temp;
    options.top_p = state->top_p;
    options.min_p = state->min_p;
    options.repeat_penalty = state->repeat_penalty;
    options.presence_penalty = state->presence_penalty;
    options.frequency_penalty = state->frequency_penalty;
    options.mirostat_tau = state->mirostat_tau;
    options.mirostat_eta = state->mirostat_eta;
    options.typical_p = state->typical_p;
    options.dry_multiplier = state->dry_multiplier;
    options.dry_base = state->dry_base;
    options.xtc_probability = state->xtc_probability;
    options.xtc_threshold = state->xtc_threshold;
    options.top_k = state->top_k;
    options.repeat_last_n = state->repeat_last_n;
    options.n_predict = state->n_predict;
    options.mirostat = state->mirostat;
    options.seed = state->seed;
    options.dry_allowed_length = state->dry_allowed_length;
    options.dry_penalty_last_n = state->dry_penalty_last_n;
    options.grammar = state->grammar ? *state->grammar : std::string();
    options.json_schema = state->json_schema ? *state->json_schema : std::string();
    options.logit_bias = state->logit_bias ? *state->logit_bias : std::vector<llama_logit_bias>();
    options.dry_sequence_breakers = state->dry_sequence_breakers ?
        *state->dry_sequence_breakers : std::vector<std::string>();
    return options;
}

static void StoreSamplingOptions(LlamaState *state, const SamplingOptions &o) {
    state->temp = o.temp;
    state->top_k = o.top_k;
    state->top_p = o.top_p;
    state->min_p = o.min_p;
    state->repeat_penalty = o.repeat_penalty;
    state->repeat_last_n = o.repeat_last_n;
    state->frequency_penalty = o.frequency_penalty;
    state->presence_penalty = o.presence_penalty;
    state->n_predict = o.n_predict;
    state->mirostat = o.mirostat;
    state->mirostat_tau = o.mirostat_tau;
    state->mirostat_eta = o.mirostat_eta;
    state->seed = o.seed;
    *state->grammar = o.grammar;
    *state->json_schema = o.json_schema;
    *state->logit_bias = o.logit_bias;
    *state->dry_sequence_breakers = o.dry_sequence_breakers;
    state->typical_p = o.typical_p;
    state->dry_multiplier = o.dry_multiplier;
    state->dry_base = o.dry_base;
    state->dry_allowed_length = o.dry_allowed_length;
    state->dry_penalty_last_n = o.dry_penalty_last_n;
    state->xtc_probability = o.xtc_probability;
    state->xtc_threshold = o.xtc_threshold;
}

static bool SameLogitBias(const std::vector<llama_logit_bias> &left,
                          const std::vector<llama_logit_bias> &right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        if (left[i].token != right[i].token || left[i].bias != right[i].bias) return false;
    }
    return true;
}

static bool SameStrings(const std::vector<std::string> &left,
                        const std::vector<std::string> &right) {
    return left == right;
}

static bool DryEnabled(const SamplingOptions &o) {
    return o.dry_multiplier > 0.0f && o.dry_penalty_last_n > 0;
}

static bool XtcEnabled(const SamplingOptions &o) {
    return o.xtc_probability > 0.0f;
}

static bool ExtraSamplersEnabled(const SamplingOptions &o) {
    return o.typical_p < 1.0f || DryEnabled(o) || XtcEnabled(o);
}

static int ParseOptionFloat(Tcl_Interp *interp, const char *name, Tcl_Obj *value,
                            float *target, double minimum, double maximum) {
    double number;
    if (Tcl_GetDoubleFromObj(interp, value, &number) != TCL_OK) return TCL_ERROR;
    if (!std::isfinite(number) || number < minimum || number > maximum) {
        Tcl_SetObjResult(interp, Tcl_ObjPrintf(
            "option \"%s\" must be between %.6g and %.6g", name, minimum, maximum));
        return TCL_ERROR;
    }
    *target = (float)number;
    return TCL_OK;
}

static int ParseOptionInt(Tcl_Interp *interp, const char *name, Tcl_Obj *value,
                          int32_t *target, int minimum, int maximum) {
    int number;
    if (Tcl_GetIntFromObj(interp, value, &number) != TCL_OK) return TCL_ERROR;
    if (number < minimum || number > maximum) {
        Tcl_SetObjResult(interp, Tcl_ObjPrintf(
            "option \"%s\" must be between %d and %d", name, minimum, maximum));
        return TCL_ERROR;
    }
    *target = (int32_t)number;
    return TCL_OK;
}

static int ParseLogitBias(Tcl_Interp *interp, Tcl_Obj *dict, int n_vocab,
                          std::vector<llama_logit_bias> *biases) {
    int count;
    Tcl_Obj **elements;
    std::vector<llama_logit_bias> parsed;
    if (Tcl_ListObjGetElements(interp, dict, &count, &elements) != TCL_OK) return TCL_ERROR;
    if (count % 2 != 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(
            "logit_bias must be a dictionary of token IDs and values", -1));
        return TCL_ERROR;
    }
    if (count / 2 > 4096 || count / 2 > n_vocab) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("logit_bias has too many entries", -1));
        return TCL_ERROR;
    }
    for (int i = 0; i < count; i += 2) {
        int token;
        if (Tcl_GetIntFromObj(interp, elements[i], &token) != TCL_OK) {
            Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                "logit_bias token ID \"%s\" must be an integer", Tcl_GetString(elements[i])));
            return TCL_ERROR;
        }
        if (token < 0 || token >= n_vocab) {
            Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                "logit_bias token ID must be between 0 and %d", n_vocab - 1));
            return TCL_ERROR;
        }
        for (size_t j = 0; j < parsed.size(); ++j) {
            if (parsed[j].token == token) {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                    "duplicate logit_bias token ID %d", token));
                return TCL_ERROR;
            }
        }
        float bias;
        if (strcmp(Tcl_GetString(elements[i + 1]), "suppress") == 0) {
            bias = -std::numeric_limits<float>::infinity();
        } else {
            double value;
            if (Tcl_GetDoubleFromObj(interp, elements[i + 1], &value) != TCL_OK) {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                    "logit_bias for token %d must be a finite number or suppress", token));
                return TCL_ERROR;
            }
            if (!std::isfinite(value) || value < -100.0 || value > 100.0) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj(
                    "finite logit_bias values must be between -100 and 100", -1));
                return TCL_ERROR;
            }
            bias = (float)value;
        }
        parsed.push_back({token, bias});
    }
    *biases = parsed;
    return TCL_OK;
}

static int ParseSamplingDict(Tcl_Interp *interp, Tcl_Obj *dict,
                             int n_vocab, SamplingOptions *options) {
    int count;
    Tcl_Obj **elements;
    std::vector<std::string> seen;
    if (Tcl_ListObjGetElements(interp, dict, &count, &elements) != TCL_OK) return TCL_ERROR;
    if (count % 2 != 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("-options value must be a key-value dictionary", -1));
        return TCL_ERROR;
    }
    for (int i = 0; i < count; i += 2) {
        Tcl_Obj *key = elements[i];
        Tcl_Obj *value = elements[i + 1];
        const char *name = Tcl_GetString(key);
        for (size_t i = 0; i < seen.size(); ++i) {
            if (seen[i] == name) {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf("duplicate option key \"%s\"", name));
                return TCL_ERROR;
            }
        }
        seen.push_back(name);
        int code = TCL_OK;
        if (strcmp(name, "grammar") == 0 || strcmp(name, "json_schema") == 0) {
            int length;
            const char *text = Tcl_GetStringFromObj(value, &length);
            if ((int)strlen(text) != length) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj("grammar must not contain NUL", -1));
                return TCL_ERROR;
            }
            if (strcmp(name, "grammar") == 0) options->grammar.assign(text, length);
            else options->json_schema.assign(text, length);
        } else if (strcmp(name, "logit_bias") == 0) {
            code = ParseLogitBias(interp, value, n_vocab, &options->logit_bias);
        } else if (strcmp(name, "typical_p") == 0)
            code = ParseOptionFloat(interp, name, value, &options->typical_p, 0.0, 1.0);
        else if (strcmp(name, "dry_multiplier") == 0)
            code = ParseOptionFloat(interp, name, value, &options->dry_multiplier, 0.0, 100.0);
        else if (strcmp(name, "dry_base") == 0)
            code = ParseOptionFloat(interp, name, value, &options->dry_base, 1.0, 100.0);
        else if (strcmp(name, "dry_allowed_length") == 0)
            code = ParseOptionInt(interp, name, value, &options->dry_allowed_length, 0, 2048);
        else if (strcmp(name, "dry_penalty_last_n") == 0)
            code = ParseOptionInt(interp, name, value, &options->dry_penalty_last_n, 0, 2048);
        else if (strcmp(name, "dry_sequence_breakers") == 0) {
            int breaker_count;
            Tcl_Obj **breaker_objects;
            std::vector<std::string> breakers;
            if (Tcl_ListObjGetElements(interp, value, &breaker_count, &breaker_objects) != TCL_OK) {
                return TCL_ERROR;
            }
            if (breaker_count > 64) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj(
                    "dry_sequence_breakers accepts at most 64 entries", -1));
                return TCL_ERROR;
            }
            for (int j = 0; j < breaker_count; ++j) {
                int length;
                const char *text = Tcl_GetStringFromObj(breaker_objects[j], &length);
                if (length == 0 || (int)strlen(text) != length || length > 40) {
                    Tcl_SetObjResult(interp, Tcl_NewStringObj(
                        "each dry_sequence_breaker must contain 1 to 40 bytes and no NUL", -1));
                    return TCL_ERROR;
                }
                std::string breaker(text, (size_t)length);
                for (size_t k = 0; k < breakers.size(); ++k) {
                    if (breakers[k] == breaker) {
                        Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                            "duplicate dry_sequence_breaker at index %d", j));
                        return TCL_ERROR;
                    }
                }
                breakers.push_back(breaker);
            }
            options->dry_sequence_breakers = breakers;
        } else if (strcmp(name, "xtc_probability") == 0)
            code = ParseOptionFloat(interp, name, value, &options->xtc_probability, 0.0, 1.0);
        else if (strcmp(name, "xtc_threshold") == 0)
            code = ParseOptionFloat(interp, name, value, &options->xtc_threshold, 0.0, 0.5);
        else if (strcmp(name, "temperature") == 0)
            code = ParseOptionFloat(interp, name, value, &options->temp, 0.0, 2.0);
        else if (strcmp(name, "top_k") == 0)
            code = ParseOptionInt(interp, name, value, &options->top_k, 1, 1000);
        else if (strcmp(name, "top_p") == 0)
            code = ParseOptionFloat(interp, name, value, &options->top_p, 0.0, 1.0);
        else if (strcmp(name, "min_p") == 0)
            code = ParseOptionFloat(interp, name, value, &options->min_p, 0.0, 1.0);
        else if (strcmp(name, "repeat_penalty") == 0)
            code = ParseOptionFloat(interp, name, value, &options->repeat_penalty, 0.000001, 2.0);
        else if (strcmp(name, "repeat_last_n") == 0)
            code = ParseOptionInt(interp, name, value, &options->repeat_last_n, 0, 2048);
        else if (strcmp(name, "frequency_penalty") == 0)
            code = ParseOptionFloat(interp, name, value, &options->frequency_penalty, -2.0, 2.0);
        else if (strcmp(name, "presence_penalty") == 0)
            code = ParseOptionFloat(interp, name, value, &options->presence_penalty, -2.0, 2.0);
        else if (strcmp(name, "num_predict") == 0)
            code = ParseOptionInt(interp, name, value, &options->n_predict, -1, INT32_MAX);
        else if (strcmp(name, "mirostat") == 0)
            code = ParseOptionInt(interp, name, value, &options->mirostat, 0, 2);
        else if (strcmp(name, "mirostat_tau") == 0)
            code = ParseOptionFloat(interp, name, value, &options->mirostat_tau, 0.0, 10.0);
        else if (strcmp(name, "mirostat_eta") == 0)
            code = ParseOptionFloat(interp, name, value, &options->mirostat_eta, 0.0, 1.0);
        else if (strcmp(name, "seed") == 0)
            code = ParseOptionInt(interp, name, value, &options->seed, -1, INT32_MAX);
        else {
            Tcl_SetObjResult(interp, Tcl_ObjPrintf("unknown sampling option \"%s\"", name));
            code = TCL_ERROR;
        }
        if (code != TCL_OK) return TCL_ERROR;
    }
    if (!options->grammar.empty() && !options->json_schema.empty()) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("grammar and json_schema are mutually exclusive", -1));
        return TCL_ERROR;
    }
    if (options->dry_multiplier > 0.0f && options->dry_penalty_last_n == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(
            "dry_multiplier requires dry_penalty_last_n greater than zero", -1));
        return TCL_ERROR;
    }
    if (options->mirostat != 0 && ExtraSamplersEnabled(*options)) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(
            "typical_p, DRY and XTC cannot be combined with Mirostat; disable them or set mirostat 0", -1));
        return TCL_ERROR;
    }
    if (options->n_predict == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(
            "option \"num_predict\" must be -1 or a positive integer", -1));
        return TCL_ERROR;
    }
    return TCL_OK;
}

static int AddSampler(struct llama_sampler *chain, struct llama_sampler *sampler) {
    if (!sampler) return 0;
    llama_sampler_chain_add(chain, sampler);
    return 1;
}

static struct llama_sampler *CreateSampler(const LlamaState *state,
                                            const SamplingOptions &o) {
    std::string effective_grammar = o.json_schema.empty() ? o.grammar : ConvertJsonSchema(o.json_schema);
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(params);
    if (!sampler) return NULL;
    if (!effective_grammar.empty() && !AddSampler(sampler,
            llama_sampler_init_grammar(state->vocab, effective_grammar.c_str(), "root"))) {
        llama_sampler_free(sampler);
        return NULL;
    }
    if (!o.logit_bias.empty() && !AddSampler(sampler,
            llama_sampler_init_logit_bias(llama_vocab_n_tokens(state->vocab),
                (int32_t)o.logit_bias.size(), o.logit_bias.data()))) {
        llama_sampler_free(sampler);
        return NULL;
    }
    if (!AddSampler(sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(state->vocab), o.repeat_last_n, o.repeat_penalty,
            o.frequency_penalty, o.presence_penalty))) {
        llama_sampler_free(sampler);
        return NULL;
    }
    if (DryEnabled(o)) {
        std::vector<const char *> breakers;
        breakers.reserve(o.dry_sequence_breakers.size());
        for (size_t i = 0; i < o.dry_sequence_breakers.size(); ++i) {
            breakers.push_back(o.dry_sequence_breakers[i].c_str());
        }
        const char **breaker_data = breakers.empty() ? NULL : breakers.data();
        if (!AddSampler(sampler, llama_sampler_init_dry(state->vocab,
                o.dry_multiplier, o.dry_base, o.dry_allowed_length,
                o.dry_penalty_last_n, breaker_data, breakers.size()))) {
            llama_sampler_free(sampler);
            return NULL;
        }
    }
    if (o.mirostat == 0) {
        if (!AddSampler(sampler, llama_sampler_init_top_k(o.top_k)) ||
            (o.typical_p < 1.0f && !AddSampler(sampler,
                llama_sampler_init_typical(o.typical_p, 1))) ||
            !AddSampler(sampler, llama_sampler_init_top_p(o.top_p, 1)) ||
            !AddSampler(sampler, llama_sampler_init_min_p(o.min_p, 1)) ||
            (XtcEnabled(o) && !AddSampler(sampler, llama_sampler_init_xtc(
                o.xtc_probability, o.xtc_threshold, 1, o.seed))) ||
            !AddSampler(sampler, llama_sampler_init_temp(o.temp)) ||
            !AddSampler(sampler, o.temp == 0.0f ? llama_sampler_init_greedy() : llama_sampler_init_dist(o.seed))) {
            llama_sampler_free(sampler);
            return NULL;
        }
    } else if (o.mirostat == 1) {
        if (!AddSampler(sampler, llama_sampler_init_temp(o.temp))) {
            llama_sampler_free(sampler);
            return NULL;
        }
        if (!AddSampler(sampler, llama_sampler_init_mirostat(
                llama_vocab_n_tokens(state->vocab), o.seed, o.mirostat_tau, o.mirostat_eta, 100))) {
            llama_sampler_free(sampler);
            return NULL;
        }
    } else if (o.mirostat == 2) {
        if (!AddSampler(sampler, llama_sampler_init_temp(o.temp))) {
            llama_sampler_free(sampler);
            return NULL;
        }
        if (!AddSampler(sampler, llama_sampler_init_mirostat_v2(
                o.seed, o.mirostat_tau, o.mirostat_eta))) {
            llama_sampler_free(sampler);
            return NULL;
        }
    }
    return sampler;
}

struct ParsedCallOptions {
    SamplingOptions sampling;
    const char *callback;
    const char *system;
    int reset;
    int has_sampling;
    std::vector<llama_token> stop_ids;
};

/* Parse all command switches before applying settings or touching model state. */
static int ParseCallOptions(Tcl_Interp *interp, int objc, Tcl_Obj *const objv[],
                            LlamaState *state, int chat, ParsedCallOptions *parsed) {
    parsed->sampling = SamplingOptionsFromState(state);
    parsed->callback = NULL;
    parsed->system = NULL;
    parsed->reset = 0;
    parsed->has_sampling = 0;
    std::vector<std::string> seen;
    int saw_max_tokens = 0;
    for (int i = 3; i < objc; i += 2) {
        const char *name = Tcl_GetString(objv[i]);
        if (i + 1 >= objc) {
            Tcl_SetObjResult(interp, Tcl_ObjPrintf("option \"%s\" requires a value", name));
            return TCL_ERROR;
        }
        for (size_t j = 0; j < seen.size(); ++j) {
            if (seen[j] == name) {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf("duplicate option \"%s\"", name));
                return TCL_ERROR;
            }
        }
        seen.push_back(name);
        if (strcmp(name, "-callback") == 0) {
            parsed->callback = Tcl_GetString(objv[i + 1]);
            if (!parsed->callback[0]) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj("option \"-callback\" must not be empty", -1));
                return TCL_ERROR;
            }
        } else if (strcmp(name, "-options") == 0) {
            if (ParseSamplingDict(interp, objv[i + 1],
                    llama_vocab_n_tokens(state->vocab), &parsed->sampling) != TCL_OK) {
                return TCL_ERROR;
            }
            parsed->has_sampling = 1;
        } else if (strcmp(name, "-reset") == 0 && !chat) {
            if (Tcl_GetBooleanFromObj(interp, objv[i + 1], &parsed->reset) != TCL_OK) {
                return TCL_ERROR;
            }
        } else if (strcmp(name, "-system") == 0 && !chat) {
            parsed->system = Tcl_GetString(objv[i + 1]);
        } else if (strcmp(name, "-max_tokens") == 0) {
            int32_t max_tokens = parsed->sampling.n_predict;
            if (ParseOptionInt(interp, name, objv[i + 1], &max_tokens, -1, INT32_MAX) != TCL_OK) {
                return TCL_ERROR;
            }
            if (max_tokens == 0) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj(
                    "option \"-max_tokens\" must be -1 or a positive integer", -1));
                return TCL_ERROR;
            }
            parsed->sampling.n_predict = max_tokens;
            parsed->has_sampling = 1;
            saw_max_tokens = 1;
        } else if (strcmp(name, "-stop_ids") == 0) {
            int count;
            Tcl_Obj **elements;
            if (Tcl_ListObjGetElements(interp, objv[i + 1], &count, &elements) != TCL_OK) {
                return TCL_ERROR;
            }
            int vocab_size = llama_vocab_n_tokens(state->vocab);
            for (int j = 0; j < count; ++j) {
                int id;
                if (Tcl_GetIntFromObj(interp, elements[j], &id) != TCL_OK) return TCL_ERROR;
                if (id < 0 || id >= vocab_size) {
                    Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                        "stop token ID must be between 0 and %d", vocab_size - 1));
                    return TCL_ERROR;
                }
                parsed->stop_ids.push_back((llama_token)id);
            }
        } else {
            Tcl_SetObjResult(interp, Tcl_ObjPrintf("unknown option \"%s\"", name));
            return TCL_ERROR;
        }
    }
    if (saw_max_tokens) {
        for (int i = 3; i < objc; i += 2) {
            if (strcmp(Tcl_GetString(objv[i]), "-options") == 0) {
                Tcl_Obj *value = NULL;
                Tcl_Obj *key = Tcl_NewStringObj("num_predict", -1);
                Tcl_IncrRefCount(key);
                int code = Tcl_DictObjGet(interp, objv[i + 1], key, &value);
                Tcl_DecrRefCount(key);
                if (code != TCL_OK) return TCL_ERROR;
                if (value) {
                    Tcl_SetObjResult(interp, Tcl_NewStringObj(
                        "-max_tokens and -options num_predict cannot be used together", -1));
                    return TCL_ERROR;
                }
            }
        }
    }
    return TCL_OK;
}

static int CommitCallSampling(Tcl_Interp *interp, LlamaState *state,
                              const ParsedCallOptions &parsed) {
    if (!parsed.has_sampling) return TCL_OK;
    const SamplingOptions &o = parsed.sampling;
    if (state->temp == o.temp && state->top_k == o.top_k && state->top_p == o.top_p &&
        state->min_p == o.min_p && state->repeat_penalty == o.repeat_penalty &&
        state->repeat_last_n == o.repeat_last_n && state->frequency_penalty == o.frequency_penalty &&
        state->presence_penalty == o.presence_penalty && state->mirostat == o.mirostat &&
        state->mirostat_tau == o.mirostat_tau && state->mirostat_eta == o.mirostat_eta &&
        state->seed == o.seed && *state->grammar == o.grammar && *state->json_schema == o.json_schema &&
        SameLogitBias(*state->logit_bias, o.logit_bias) &&
        state->typical_p == o.typical_p && state->dry_multiplier == o.dry_multiplier &&
        state->dry_base == o.dry_base && state->dry_allowed_length == o.dry_allowed_length &&
        state->dry_penalty_last_n == o.dry_penalty_last_n &&
        state->xtc_probability == o.xtc_probability && state->xtc_threshold == o.xtc_threshold &&
        SameStrings(*state->dry_sequence_breakers, o.dry_sequence_breakers)) {
        state->n_predict = o.n_predict;
        return TCL_OK;
    }
    struct llama_sampler *sampler;
    try { sampler = CreateSampler(state, parsed.sampling); }
    catch (const std::exception &e) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj(e.what(), -1));
        return TCL_ERROR;
    }
    if (!sampler) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create sampler or invalid grammar", -1));
        return TCL_ERROR;
    }
    struct llama_sampler *previous = state->sampler;
    state->sampler = sampler;
    StoreSamplingOptions(state, parsed.sampling);
    if (previous) llama_sampler_free(previous);
    return TCL_OK;
}

static void fill_batch(struct llama_batch & batch, llama_token id, int pos, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens]   = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

/* Discard inference history while retaining the handle's configured options. */
static void ResetInferenceState(LlamaState *state) {
    llama_memory_clear(llama_get_memory(state->ctx), true);
    llama_sampler_reset(state->sampler);
    state->n_past = 0;
    state->sequence_fork_pending = false;
    if (state->sequence_positions) {
        for (int &position : *state->sequence_positions) if (position >= 0) position = 0;
    }
    if (state->sequence_sampling) {
        for (SequenceSampling *s : *state->sequence_sampling) if (s) {
            if (s->sampler) llama_sampler_reset(s->sampler);
            s->logits.clear(); s->started = s->finished = false;
        }
    }
}

static int RequireSingleSequence(Tcl_Interp *interp, LlamaState *state) {
    if (state->sequence_positions) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Operation unavailable in multi-sequence prefill mode", -1));
        return TCL_ERROR;
    }
    return TCL_OK;
}

/* Decode contiguous tokens in sequence 0; caller validates capacity and owns recovery. */
static int DecodePromptBatch(LlamaState *state, const llama_token *tokens, int count, int past) {
    struct llama_batch batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) {
        fill_batch(batch, tokens[i], past + i, i == count - 1);
    }
    int result = llama_decode(state->ctx, batch);
    llama_batch_free(batch);
    return result;
}

/* Append explicit token IDs as prompt input to the only supported sequence. */
static int Llama_Prefill_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 3) { Tcl_WrongNumArgs(interp, 1, objv, "handle tokenIds"); return TCL_ERROR; }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (RequireSingleSequence(interp, state) != TCL_OK) return TCL_ERROR;
    int count = 0;
    Tcl_Obj **items = NULL;
    if (Tcl_ListObjGetElements(interp, objv[2], &count, &items) != TCL_OK) return TCL_ERROR;
    if (count <= 0 || (uint32_t)count > llama_n_batch(state->ctx)) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("token list must contain between 1 and n_batch tokens", -1)); return TCL_ERROR;
    }
    if (count >= state->n_ctx - state->n_past) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("token list exceeds remaining context capacity", -1)); return TCL_ERROR;
    }
    std::vector<llama_token> tokens;
    tokens.reserve(count);
    for (int i = 0; i < count; ++i) {
        int id;
        if (Tcl_GetIntFromObj(interp, items[i], &id) != TCL_OK) return TCL_ERROR;
        if (id < 0 || id >= llama_vocab_n_tokens(state->vocab)) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid token ID", -1)); return TCL_ERROR;
        }
        tokens.push_back(id);
    }
    LlamaOperation operation(state);
    if (DecodePromptBatch(state, tokens.data(), count, state->n_past) != 0) {
        ResetInferenceState(state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed; inference state reset", -1)); return TCL_ERROR;
    }
    state->n_past += count;
    Tcl_SetObjResult(interp, Tcl_NewIntObj(state->n_past));
    return TCL_OK;
}

/* Deliver exactly the same ordered bytes to the result and the callback. */
static int DeliverText(Tcl_Interp *interp, LlamaState *state, const char *callback,
                       Tcl_DString *response, const std::string &text) {
    if (text.empty()) return TCL_OK;
    Tcl_DStringAppend(response, text.data(), (int)text.size());
    if (callback && InvokeCallback(interp, state, callback, text.data(), (int)text.size()) != TCL_OK) {
        ResetInferenceState(state);
        return TCL_ERROR;
    }
    return TCL_OK;
}

/* Restricted sampling validates each intermediate candidate set before the
 * native selector accepts a token. This also protects logit-bias suppression. */
static int SampleFromLogits(Tcl_Interp *interp, llama_sampler *sampler, int n,
                            const float *logits, llama_token *out) {
    if (!logits || n <= 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Sampling logits unavailable", -1));
        return TCL_ERROR;
    }
    try {
        std::vector<llama_token_data> data;
        data.reserve(n);
        for (int i = 0; i < n; ++i) data.push_back({i, logits[i], 0.0f});
        llama_token_data_array candidates = {data.data(), data.size(), -1, false};
        int count = llama_sampler_chain_n(sampler);
        for (int i = 0; i < count; ++i) {
            // Keep a stray NaN or positive infinity out of native selectors.
            for (size_t j = 0; j < candidates.size; ++j) {
                if (std::isnan(candidates.data[j].logit) ||
                    candidates.data[j].logit == std::numeric_limits<float>::infinity()) {
                    candidates.data[j].logit = -std::numeric_limits<float>::infinity();
                    candidates.sorted = false;
                }
            }
            llama_sampler_apply(llama_sampler_chain_get(sampler, i), &candidates);
            if (!HasSelectableCandidate(candidates)) {
                Tcl_SetObjResult(interp, Tcl_NewStringObj("Sampling has no selectable candidates", -1));
                return TCL_ERROR;
            }
        }
        if (!HasValidSelection(candidates)) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Sampling selected an invalid candidate", -1));
            return TCL_ERROR;
        }
        *out = candidates.data[candidates.selected].id;
        llama_sampler_accept(sampler, *out);
        return TCL_OK;
    } catch (const std::exception &e) {
        Tcl_SetObjResult(interp, Tcl_ObjPrintf("Sampling failed: %s", e.what()));
        return TCL_ERROR;
    }
}

static int SampleWithValidatedChain(Tcl_Interp *interp, LlamaState *state, llama_token *out) {
    return SampleFromLogits(interp, state->sampler, llama_vocab_n_tokens(state->vocab),
                            llama_get_logits_ith(state->ctx, -1), out);
}

/* ----------------- CORE GENERATION LOOP (v7.5 - Universal + Buffer) ----------------- */
static int run_inference(Tcl_Interp *interp, LlamaState *state, const char *cb_name, 
                        std::vector<llama_token> & stop_ids) {
    Tcl_DString resp;
    Tcl_DStringInit(&resp);
    
    const bool restricted_sampling = !state->grammar->empty() ||
        !state->json_schema->empty() || !state->logit_bias->empty() ||
        ExtraSamplersEnabled(SamplingOptionsFromState(state));
    TextBuffer text_buffer(state->grammar->empty() && state->json_schema->empty());
    
    auto t_start_gen = std::chrono::high_resolution_clock::now();
    
    int max_tokens = (state->n_predict > 0) ? state->n_predict : 4096;
    int p_cnt = 0;
    
    while (p_cnt < max_tokens) {
        if (state->n_past >= state->n_ctx) break;
        
        llama_token id;
        if (!restricted_sampling) id = llama_sampler_sample(state->sampler, state->ctx, -1);
        else if (SampleWithValidatedChain(interp, state, &id) != TCL_OK) {
            ResetInferenceState(state);
            Tcl_DStringFree(&resp);
            return TCL_ERROR;
        }
        
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
        
        char small_piece[512];
        int n = llama_token_to_piece(state->vocab, id, small_piece, sizeof(small_piece), 0, false);
        std::vector<char> large_piece;
        const char *piece = small_piece;
        if (n < 0) {
            large_piece.resize((size_t)-n);
            n = llama_token_to_piece(state->vocab, id, large_piece.data(), large_piece.size(), 0, false);
            piece = large_piece.data();
        }
        if (n < 0) {
            ResetInferenceState(state);
            Tcl_DStringFree(&resp);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Token conversion failed", -1));
            return TCL_ERROR;
        }
        if (n > 0 && DeliverText(interp, state, cb_name, &resp,
                                 text_buffer.push(piece, (size_t)n)) != TCL_OK) {
            Tcl_DStringFree(&resp);
            return TCL_ERROR;
        }
        if (text_buffer.stopped()) break;

        struct llama_batch b = llama_batch_init(1, 0, 1);
        fill_batch(b, id, state->n_past, true);
        
        if (llama_decode(state->ctx, b) != 0) {
            llama_batch_free(b);
            Tcl_DStringFree(&resp);
            ResetInferenceState(state);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed during generation", -1));
            return TCL_ERROR;
        }
        llama_batch_free(b);
        state->n_past++;
        p_cnt++;
    }
    
    if (DeliverText(interp, state, cb_name, &resp, text_buffer.finish()) != TCL_OK) {
        Tcl_DStringFree(&resp);
        return TCL_ERROR;
    }

    auto t_end_gen = std::chrono::high_resolution_clock::now();
    state->t_gen_ms = std::chrono::duration<double, std::milli>(t_end_gen - t_start_gen).count();
    state->n_gen = p_cnt;
    
    Tcl_SetObjResult(interp, Tcl_NewStringObj(Tcl_DStringValue(&resp), Tcl_DStringLength(&resp)));
    Tcl_DStringFree(&resp);
    return TCL_OK;
}

/* ----------------- LLAMA::GENERATE (Stateful) ----------------- */
static int Llama_Generate_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::generate handle prompt ?-callback proc? ?-options dict? ?-reset bool? ?-stop_ids list? ?-system string? ?-max_tokens int?", -1));
        return TCL_ERROR;
    }
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (RequireSingleSequence(interp, state) != TCL_OK) return TCL_ERROR;
    LlamaOperation operation(state);

    const char *prompt = Tcl_GetString(objv[2]);
    ParsedCallOptions parsed;
    if (ParseCallOptions(interp, objc, objv, state, 0, &parsed) != TCL_OK) return TCL_ERROR;
    const int past = parsed.reset ? 0 : state->n_past;
    
    // Construir prompt con system message si existe
    std::string full_prompt;
    if (parsed.system && parsed.system[0] && past == 0) {
        full_prompt = std::string(parsed.system) + "\n\n" + std::string(prompt);
    } else {
        full_prompt = std::string(prompt);
    }
    
    std::vector<llama_token> tokens(full_prompt.length() + 256);
    int n_tok = llama_tokenize(state->vocab, full_prompt.c_str(), full_prompt.length(), 
                                tokens.data(), tokens.size(), (past == 0), false);
    
    if (n_tok < 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Tokenization failed", -1));
        return TCL_ERROR;
    }

    // Verificar overflow de contexto
    if (past + n_tok >= state->n_ctx) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Context overflow: n_past=%d + n_tok=%d >= n_ctx=%d", 
                 past, n_tok, state->n_ctx);
        Tcl_SetObjResult(interp, Tcl_NewStringObj(msg, -1));
        return TCL_ERROR;
    }

    if (CommitCallSampling(interp, state, parsed) != TCL_OK) return TCL_ERROR;
    if (parsed.reset) {
        ResetInferenceState(state);
    }

    // Telemetría: medir tiempo de ingestión del prompt
    auto t_start_eval = std::chrono::high_resolution_clock::now();
    
    if (DecodePromptBatch(state, tokens.data(), n_tok, past) != 0) {
        ResetInferenceState(state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed", -1));
        return TCL_ERROR;
    }
    state->n_past = past + n_tok;
    
    auto t_end_eval = std::chrono::high_resolution_clock::now();
    state->t_eval_ms = std::chrono::duration<double, std::milli>(t_end_eval - t_start_eval).count();
    state->n_eval = n_tok;

    return run_inference(interp, state, parsed.callback, parsed.stop_ids);
}

/* ----------------- LLAMA::CHAT (Stateless) ----------------- */
static int Llama_Chat(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::chat handle messages ?-callback proc? ?-options dict? ?-stop_ids list? ?-max_tokens int?", -1));
        return TCL_ERROR;
    }
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (RequireSingleSequence(interp, state) != TCL_OK) return TCL_ERROR;
    LlamaOperation operation(state);

    int n_msgs;
    Tcl_Obj **msgs_elems;
    if (Tcl_ListObjGetElements(interp, objv[2], &n_msgs, &msgs_elems) != TCL_OK) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid messages list", -1));
        return TCL_ERROR;
    }
    if (n_msgs == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("messages must contain at least one message", -1));
        return TCL_ERROR;
    }

    ParsedCallOptions parsed;
    if (ParseCallOptions(interp, objc, objv, state, 1, &parsed) != TCL_OK) return TCL_ERROR;
    
    std::vector<llama_chat_message> cmsgs;
    std::vector<char*> allocated_strings;
    
    for (int i = 0; i < n_msgs; i++) {
        Tcl_Obj *v_role, *v_content;
        Tcl_Obj *role_key = Tcl_NewStringObj("role", -1);
        Tcl_Obj *content_key = Tcl_NewStringObj("content", -1);
        
        Tcl_IncrRefCount(role_key);
        Tcl_IncrRefCount(content_key);
        
        int role_code = Tcl_DictObjGet(interp, msgs_elems[i], role_key, &v_role);
        int content_code = Tcl_DictObjGet(interp, msgs_elems[i], content_key, &v_content);
        
        Tcl_DecrRefCount(role_key);
        Tcl_DecrRefCount(content_key);
        
        if (role_code != TCL_OK || content_code != TCL_OK || !v_role || !v_content ||
            Tcl_GetCharLength(v_role) == 0) {
            for (auto ptr : allocated_strings) free(ptr);
            Tcl_SetObjResult(interp, Tcl_ObjPrintf(
                "message %d must be a dict with non-empty role and content fields", i));
            return TCL_ERROR;
        }
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
                                tokens.data(), tokens.size(), true, true);
    
    if (n_tok < 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Tokenization failed", -1));
        return TCL_ERROR;
    }

    if (n_tok >= state->n_ctx) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Context overflow in chat", -1));
        return TCL_ERROR;
    }
    if (CommitCallSampling(interp, state, parsed) != TCL_OK) return TCL_ERROR;
    // Chat has always used an independent context; clear only after all input
    // parsing, template preparation and tokenization have succeeded.
    ResetInferenceState(state);

    // Telemetría: medir tiempo de ingestión
    auto t_start_eval = std::chrono::high_resolution_clock::now();
    
    struct llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; i++) {
        fill_batch(batch, tokens[i], i, (i == n_tok - 1));
    }
    
    if (llama_decode(state->ctx, batch) != 0) {
        llama_batch_free(batch);
        ResetInferenceState(state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed", -1));
        return TCL_ERROR;
    }
    llama_batch_free(batch);
    state->n_past = n_tok;
    
    auto t_end_eval = std::chrono::high_resolution_clock::now();
    state->t_eval_ms = std::chrono::duration<double, std::milli>(t_end_eval - t_start_eval).count();
    state->n_eval = n_tok;

    return run_inference(interp, state, parsed.callback, parsed.stop_ids);
}

/* ----------------- LLAMA::DETOKENIZE (v7.0) ----------------- */
static int Llama_Detokenize_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::detokenize handle token_list", -1));
        return TCL_ERROR;
    }
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    
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

/* Add an optional scalar GGUF value without manufacturing a fallback value. */
static void PutModelMeta(Tcl_Interp *interp, Tcl_Obj *dict,
                         const struct llama_model *model,
                         const char *field, const char *metadata_key) {
    char small[256];
    int32_t length = llama_model_meta_val_str(model, metadata_key, small, sizeof(small));
    if (length < 0) return;
    Tcl_Obj *value;
    if ((size_t)length < sizeof(small)) {
        value = Tcl_NewStringObj(small, length);
    } else {
        std::vector<char> buffer((size_t)length + 1);
        int32_t actual = llama_model_meta_val_str(model, metadata_key, buffer.data(), buffer.size());
        if (actual < 0) return;
        value = Tcl_NewStringObj(buffer.data(), actual);
    }
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj(field, -1), value);
}

static const char *VocabTypeName(enum llama_vocab_type type) {
    switch (type) {
        case LLAMA_VOCAB_TYPE_NONE: return "none";
        case LLAMA_VOCAB_TYPE_SPM: return "spm";
        case LLAMA_VOCAB_TYPE_BPE: return "bpe";
        case LLAMA_VOCAB_TYPE_WPM: return "wpm";
        case LLAMA_VOCAB_TYPE_UGM: return "ugm";
        case LLAMA_VOCAB_TYPE_RWKV: return "rwkv";
        case LLAMA_VOCAB_TYPE_PLAMO2: return "plamo2";
        case LLAMA_VOCAB_TYPE_TEST: return "test";
        case LLAMA_VOCAB_TYPE_PLAMO3: return "plamo3";
    }
    return NULL;
}

/* ----------------- LLAMA::INFO - Model metadata and effective context ----------------- */
static int Llama_Info_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::info handle", -1));
        return TCL_ERROR;
    }
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    
    Tcl_Obj *dict = Tcl_NewDictObj();
    
    // Información del contexto (v6.9)
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_seq_max", -1), Tcl_NewIntObj(llama_n_seq_max(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx_seq", -1), Tcl_NewIntObj(llama_n_ctx_seq(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("sequence_mode", -1),
                   Tcl_NewStringObj(state->sequence_generation ? "independent" :
                       (state->sequence_positions ? "prefill_only" : "single"), -1));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx", -1), 
                   Tcl_NewIntObj(llama_n_ctx(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_past", -1),
                   Tcl_NewIntObj(state->n_past));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx_used", -1),
                   Tcl_NewIntObj(state->n_past));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ctx_available", -1),
                   Tcl_NewIntObj(state->n_ctx - state->n_past));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_batch", -1),
                   Tcl_NewIntObj(llama_n_batch(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_ubatch", -1),
                   Tcl_NewIntObj(llama_n_ubatch(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_threads", -1),
                   Tcl_NewIntObj(llama_n_threads(state->ctx)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("n_threads_batch", -1),
                   Tcl_NewIntObj(llama_n_threads_batch(state->ctx)));
    
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

    Tcl_Obj *model_info = Tcl_NewDictObj();
    PutModelMeta(interp, model_info, state->model, "architecture", "general.architecture");
    PutModelMeta(interp, model_info, state->model, "name", "general.name");
    PutModelMeta(interp, model_info, state->model, "description", "general.description");
    PutModelMeta(interp, model_info, state->model, "basename", "general.basename");
    PutModelMeta(interp, model_info, state->model, "finetune", "general.finetune");
    PutModelMeta(interp, model_info, state->model, "size_label", "general.size_label");
    PutModelMeta(interp, model_info, state->model, "tokenizer_model", "tokenizer.ggml.model");
    PutModelMeta(interp, model_info, state->model, "tokenizer_pre", "tokenizer.ggml.pre");
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("file_type", -1),
                   Tcl_NewStringObj(llama_ftype_name(llama_model_ftype(state->model)), -1));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_ctx_train", -1),
                   Tcl_NewIntObj(llama_model_n_ctx_train(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_embd", -1),
                   Tcl_NewIntObj(llama_model_n_embd(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_embd_inp", -1),
                   Tcl_NewIntObj(llama_model_n_embd_inp(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_embd_out", -1),
                   Tcl_NewIntObj(llama_model_n_embd_out(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_layer", -1),
                   Tcl_NewIntObj(llama_model_n_layer(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_head", -1),
                   Tcl_NewIntObj(llama_model_n_head(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("n_head_kv", -1),
                   Tcl_NewIntObj(llama_model_n_head_kv(state->model)));
    const char *vocab_type = VocabTypeName(llama_vocab_type(state->vocab));
    if (vocab_type) {
        Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("vocab_type", -1),
                       Tcl_NewStringObj(vocab_type, -1));
    }
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("has_encoder", -1),
                   Tcl_NewBooleanObj(llama_model_has_encoder(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("has_decoder", -1),
                   Tcl_NewBooleanObj(llama_model_has_decoder(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("is_recurrent", -1),
                   Tcl_NewBooleanObj(llama_model_is_recurrent(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("is_hybrid", -1),
                   Tcl_NewBooleanObj(llama_model_is_hybrid(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("is_diffusion", -1),
                   Tcl_NewBooleanObj(llama_model_is_diffusion(state->model)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("token_bos", -1),
                   Tcl_NewIntObj(llama_vocab_bos(state->vocab)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("token_eos", -1),
                   Tcl_NewIntObj(llama_vocab_eos(state->vocab)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("token_eot", -1),
                   Tcl_NewIntObj(llama_vocab_eot(state->vocab)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("tokenizer_add_bos", -1),
                   Tcl_NewBooleanObj(llama_vocab_get_add_bos(state->vocab)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("tokenizer_add_eos", -1),
                   Tcl_NewBooleanObj(llama_vocab_get_add_eos(state->vocab)));
    Tcl_DictObjPut(interp, model_info, Tcl_NewStringObj("tokenizer_add_sep", -1),
                   Tcl_NewBooleanObj(llama_vocab_get_add_sep(state->vocab)));
    Tcl_DictObjPut(interp, dict, Tcl_NewStringObj("model", -1), model_info);
    
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
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("typical_p", -1),
                   Tcl_NewDoubleObj(state->typical_p));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("repeat_penalty", -1),
                   Tcl_NewDoubleObj(state->repeat_penalty));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("dry_multiplier", -1),
                   Tcl_NewDoubleObj(state->dry_multiplier));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("dry_base", -1),
                   Tcl_NewDoubleObj(state->dry_base));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("dry_allowed_length", -1),
                   Tcl_NewIntObj(state->dry_allowed_length));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("dry_penalty_last_n", -1),
                   Tcl_NewIntObj(state->dry_penalty_last_n));
    Tcl_Obj *breakers = Tcl_NewListObj(0, NULL);
    for (size_t i = 0; i < state->dry_sequence_breakers->size(); ++i) {
        const std::string &breaker = (*state->dry_sequence_breakers)[i];
        Tcl_ListObjAppendElement(interp, breakers,
            Tcl_NewStringObj(breaker.data(), (int)breaker.size()));
    }
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("dry_sequence_breakers", -1), breakers);
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("xtc_probability", -1),
                   Tcl_NewDoubleObj(state->xtc_probability));
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("xtc_threshold", -1),
                   Tcl_NewDoubleObj(state->xtc_threshold));
    Tcl_Obj *bias_info = Tcl_NewDictObj();
    for (size_t i = 0; i < state->logit_bias->size(); ++i) {
        const llama_logit_bias &entry = (*state->logit_bias)[i];
        Tcl_Obj *value = std::isinf(entry.bias) ? Tcl_NewStringObj("suppress", -1) :
            Tcl_NewDoubleObj(entry.bias);
        Tcl_DictObjPut(interp, bias_info, Tcl_NewIntObj(entry.token), value);
    }
    Tcl_DictObjPut(interp, sampling, Tcl_NewStringObj("logit_bias", -1), bias_info);
    
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
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    
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
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    
    Tcl_SetObjResult(interp, Tcl_NewIntObj(state->n_past));
    return TCL_OK;
}

/* Inspect the single sequence used by this binding; positions are native KV positions. */
static int Llama_MemoryInfo_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 2) { Tcl_WrongNumArgs(interp, 1, objv, "handle"); return TCL_ERROR; }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    llama_memory_t mem = llama_get_memory(state->ctx);
    if (!mem) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Model has no inference memory", -1)); return TCL_ERROR; }
    Tcl_Obj *result = Tcl_NewDictObj();
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("sequence", -1), Tcl_NewIntObj(0));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("pos_min", -1), Tcl_NewIntObj(llama_memory_seq_pos_min(mem, 0)));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("pos_max", -1), Tcl_NewIntObj(llama_memory_seq_pos_max(mem, 0)));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("n_past", -1), Tcl_NewIntObj(state->n_past));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("can_shift", -1), Tcl_NewBooleanObj(llama_memory_can_shift(mem)));
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
}

/* Truncate a suffix in a temporary native context, preserving active state on failure. */
static int Llama_MemoryTruncate_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 3) { Tcl_WrongNumArgs(interp, 1, objv, "handle position"); return TCL_ERROR; }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (RequireSingleSequence(interp, state) != TCL_OK) return TCL_ERROR;
    int position;
    if (Tcl_GetIntFromObj(interp, objv[2], &position) != TCL_OK) return TCL_ERROR;
    if (position < 0 || position > state->n_past) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("position must be between 0 and n_past", -1)); return TCL_ERROR;
    }
    if (position == state->n_past) { Tcl_SetObjResult(interp, Tcl_NewIntObj(position)); return TCL_OK; }
    llama_memory_t active = llama_get_memory(state->ctx);
    if (!active) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Model has no inference memory", -1)); return TCL_ERROR; }
    int min = llama_memory_seq_pos_min(active, 0);
    int max = llama_memory_seq_pos_max(active, 0);
    if (position && (min != 0 || max != state->n_past - 1)) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Cannot truncate non-contiguous or windowed memory", -1)); return TCL_ERROR;
    }
    LlamaOperation operation(state);
    size_t size = llama_state_get_size(state->ctx);
    unsigned char *data = size ? (unsigned char *)malloc(size) : NULL;
    if (!data) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Cannot allocate temporary context state", -1)); return TCL_ERROR; }
    struct llama_context *context = NULL;
    if (llama_state_get_data(state->ctx, data, size) == size) context = llama_init_from_model(state->model, state->context_params);
    bool ok = context && llama_n_ctx(context) == llama_n_ctx(state->ctx) &&
        llama_state_set_data(context, data, size) == size;
    free(data);
    llama_memory_t memory = ok ? llama_get_memory(context) : NULL;
    if (memory) ok = llama_memory_seq_rm(memory, 0, position, -1);
    else ok = false;
    if (!ok) {
        if (context) llama_free(context);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Cannot truncate this model's inference memory", -1)); return TCL_ERROR;
    }
    struct llama_context *previous = state->ctx;
    state->ctx = context;
    state->n_past = position;
    llama_sampler_reset(state->sampler);
    state->t_eval_ms = state->t_gen_ms = 0.0;
    state->n_eval = state->n_gen = 0;
    llama_free(previous);
    Tcl_SetObjResult(interp, Tcl_NewIntObj(position));
    return TCL_OK;
}

/* ----------------- SOPORTE E INIT ----------------- */
static int Llama_Tokenize_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 3) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::tokenize handle text", -1));
        return TCL_ERROR;
    }
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    
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
    
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    
    ResetInferenceState(state);
    
    // Reset telemetría
    state->t_eval_ms = 0.0;
    state->t_gen_ms = 0.0;
    state->n_eval = 0;
    state->n_gen = 0;
    
    return TCL_OK;
}

/* Each command operates on the owning handle's single private snapshot. */
static int Llama_SnapshotSave_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 2) { Tcl_WrongNumArgs(interp, 1, objv, "handle"); return TCL_ERROR; }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (RequireSingleSequence(interp, state) != TCL_OK) return TCL_ERROR;
    LlamaOperation operation(state);
    size_t size = llama_state_get_size(state->ctx);
    if (!size || size > (size_t)std::numeric_limits<Tcl_WideInt>::max()) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Cannot determine snapshot size", -1));
        return TCL_ERROR;
    }
    InferenceSnapshot *snapshot = new (std::nothrow) InferenceSnapshot();
    if (!snapshot) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Snapshot allocation failed", -1)); return TCL_ERROR; }
    snapshot->data = (uint8_t *)malloc(size);
    snapshot->size = size;
    snapshot->sampler = NULL;
    if (!snapshot->data || llama_state_get_data(state->ctx, snapshot->data, size) != size ||
        !(snapshot->sampler = llama_sampler_clone(state->sampler))) {
        FreeSnapshot(snapshot);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to save snapshot", -1));
        return TCL_ERROR;
    }
    snapshot->n_past = state->n_past;
    snapshot->options = SamplingOptionsFromState(state);
    InferenceSnapshot *previous = state->snapshot;
    state->snapshot = snapshot;
    FreeSnapshot(previous);
    Tcl_SetObjResult(interp, Tcl_NewWideIntObj((Tcl_WideInt)size));
    return TCL_OK;
}

static int Llama_SnapshotRestore_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 2) { Tcl_WrongNumArgs(interp, 1, objv, "handle"); return TCL_ERROR; }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (RequireSingleSequence(interp, state) != TCL_OK) return TCL_ERROR;
    LlamaOperation operation(state);
    InferenceSnapshot *snapshot = state->snapshot;
    if (!snapshot) { Tcl_SetObjResult(interp, Tcl_NewStringObj("No snapshot saved for handle", -1)); return TCL_ERROR; }
    if (!snapshot->data || !snapshot->size || snapshot->n_past < 0 || snapshot->n_past > state->n_ctx) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid snapshot", -1)); return TCL_ERROR;
    }
    struct llama_sampler *sampler = llama_sampler_clone(snapshot->sampler);
    if (!sampler) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to clone snapshot sampler", -1)); return TCL_ERROR; }
    struct llama_context *context = llama_init_from_model(state->model, state->context_params);
    if (!context) {
        llama_sampler_free(sampler);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create restore context", -1)); return TCL_ERROR;
    }
    if (llama_n_ctx(context) != llama_n_ctx(state->ctx) ||
        llama_state_set_data(context, snapshot->data, snapshot->size) != snapshot->size) {
        llama_free(context);
        llama_sampler_free(sampler);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to restore snapshot", -1)); return TCL_ERROR;
    }
    struct llama_context *old_context = state->ctx;
    struct llama_sampler *old_sampler = state->sampler;
    state->ctx = context;
    state->sampler = sampler;
    state->n_past = snapshot->n_past;
    StoreSamplingOptions(state, snapshot->options);
    state->t_eval_ms = state->t_gen_ms = 0.0;
    state->n_eval = state->n_gen = 0;
    llama_sampler_free(old_sampler);
    llama_free(old_context);
    return TCL_OK;
}

static int Llama_SnapshotDiscard_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 2) { Tcl_WrongNumArgs(interp, 1, objv, "handle"); return TCL_ERROR; }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (!state->snapshot) { Tcl_SetObjResult(interp, Tcl_NewStringObj("No snapshot saved for handle", -1)); return TCL_ERROR; }
    FreeSnapshot(state->snapshot);
    state->snapshot = NULL;
    return TCL_OK;
}

struct InitOptions {
    int sequence_sampling;
    int n_seq_max;
    int n_threads;
    int n_threads_batch;
    int n_batch;
    int n_ubatch;
    int n_gpu_layers;
    int main_gpu;
    int check_tensors;
    int offload_kqv;
    int flash_attn;
    InitOptions() : sequence_sampling(0), n_seq_max(1), n_threads(0), n_threads_batch(0), n_batch(2048),
        n_ubatch(0), n_gpu_layers(0), main_gpu(0), check_tensors(-1),
        offload_kqv(-1), flash_attn(-1) {}
};

static int ParseInitOptions(Tcl_Interp *interp, Tcl_Obj *obj, InitOptions *options) {
    int count;
    Tcl_Obj **items;
    std::vector<std::string> seen;
    if (Tcl_ListObjGetElements(interp, obj, &count, &items) != TCL_OK) return TCL_ERROR;
    if (count % 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("-options must be a key-value dictionary", -1));
        return TCL_ERROR;
    }
    for (int i = 0; i < count; i += 2) {
        const char *key = Tcl_GetString(items[i]);
        for (size_t j = 0; j < seen.size(); ++j) {
            if (seen[j] == key) {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf("duplicate initialization option \"%s\"", key));
                return TCL_ERROR;
            }
        }
        seen.push_back(key);
        int value;
        if (strcmp(key, "check_tensors") == 0 || strcmp(key, "offload_kqv") == 0 ||
            strcmp(key, "flash_attn") == 0 || strcmp(key, "sequence_sampling") == 0) {
            if (Tcl_GetBooleanFromObj(interp, items[i + 1], &value) != TCL_OK) return TCL_ERROR;
            if (strcmp(key, "check_tensors") == 0) options->check_tensors = value;
            else if (strcmp(key, "offload_kqv") == 0) options->offload_kqv = value;
            else if (strcmp(key, "sequence_sampling") == 0) options->sequence_sampling = value;
            else options->flash_attn = value;
        } else {
            int minimum = 1;
            int maximum = 2048;
            int32_t *target = NULL;
            if (strcmp(key, "n_threads") == 0) { maximum = 256; target = (int32_t *)&options->n_threads; }
            else if (strcmp(key, "n_threads_batch") == 0) { maximum = 256; target = (int32_t *)&options->n_threads_batch; }
            else if (strcmp(key, "n_batch") == 0) target = (int32_t *)&options->n_batch;
            else if (strcmp(key, "n_seq_max") == 0) { maximum = 64; target = (int32_t *)&options->n_seq_max; }
            else if (strcmp(key, "n_ubatch") == 0) target = (int32_t *)&options->n_ubatch;
            else if (strcmp(key, "n_gpu_layers") == 0) { minimum = -1; maximum = 999; target = (int32_t *)&options->n_gpu_layers; }
            else if (strcmp(key, "main_gpu") == 0) { minimum = 0; maximum = 10; target = (int32_t *)&options->main_gpu; }
            else {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf("unknown initialization option \"%s\"", key));
                return TCL_ERROR;
            }
            if (Tcl_GetIntFromObj(interp, items[i + 1], &value) != TCL_OK) return TCL_ERROR;
            if (value < minimum || value > maximum) {
                Tcl_SetObjResult(interp, Tcl_ObjPrintf("option \"%s\" must be between %d and %d", key, minimum, maximum));
                return TCL_ERROR;
            }
            *target = value;
        }
    }
    if (options->n_ubatch && options->n_ubatch > options->n_batch) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("n_ubatch cannot exceed n_batch", -1));
        return TCL_ERROR;
    }
    if (options->sequence_sampling && options->n_seq_max == 1) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("sequence_sampling requires n_seq_max greater than 1", -1)); return TCL_ERROR;
    }
    return TCL_OK;
}

static int Llama_Init_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc < 2 || objc > 5) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::init model_path ?n_ctx? ?-options dict?", -1));
        return TCL_ERROR;
    }
    
    const char *model_path = Tcl_GetString(objv[1]);
    if (!model_path[0]) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("model_path must not be empty", -1));
        return TCL_ERROR;
    }
    int n_ctx = 4096;
    
    InitOptions init_options;
    Tcl_Obj *options_obj = NULL;
    int next_arg = 2;
    if (next_arg < objc && strcmp(Tcl_GetString(objv[next_arg]), "-options") != 0) {
        if (Tcl_GetIntFromObj(interp, objv[2], &n_ctx) != TCL_OK) {
            return TCL_ERROR;
        }
        if (n_ctx < 512 || n_ctx > 32768) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("n_ctx must be between 512 and 32768", -1));
            return TCL_ERROR;
        }
        ++next_arg;
    }
    if (next_arg < objc) {
        if (strcmp(Tcl_GetString(objv[next_arg]), "-options") != 0 || next_arg + 2 != objc) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("expected ?-options dict? after n_ctx", -1));
            return TCL_ERROR;
        }
        options_obj = objv[next_arg + 1];
        if (ParseInitOptions(interp, options_obj, &init_options) != TCL_OK) return TCL_ERROR;
    }
    
    LlamaState *state = (LlamaState*)ckalloc(sizeof(LlamaState));
    if (!state) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Memory allocation failed", -1));
        return TCL_ERROR;
    }
    
    memset(state, 0, sizeof(LlamaState));
    state->grammar = new (std::nothrow) std::string();
    state->json_schema = new (std::nothrow) std::string();
    state->logit_bias = new (std::nothrow) std::vector<llama_logit_bias>();
    state->dry_sequence_breakers = new (std::nothrow) std::vector<std::string>();
    if (!state->grammar || !state->json_schema || !state->logit_bias ||
        !state->dry_sequence_breakers) {
        LlamaStateFree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Memory allocation failed", -1));
        return TCL_ERROR;
    }
    *state->dry_sequence_breakers = {"\n", ":", "\"", "*"};
    state->interp = interp;
    set_defaults(state);
    state->n_ctx = n_ctx;
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = init_options.n_gpu_layers;
    mparams.main_gpu = init_options.main_gpu;
    if (init_options.check_tensors >= 0) mparams.check_tensors = init_options.check_tensors;
    state->model = llama_model_load_from_file(model_path, mparams);
    
    if (!state->model) {
        LlamaStateFree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to load model", -1));
        return TCL_ERROR;
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_seq_max = init_options.n_seq_max;
    cparams.n_batch = init_options.n_batch;
    if (init_options.n_ubatch) cparams.n_ubatch = init_options.n_ubatch;
    else if (cparams.n_ubatch > cparams.n_batch) cparams.n_ubatch = cparams.n_batch;
    if (init_options.n_threads) cparams.n_threads = init_options.n_threads;
    if (init_options.n_threads_batch) cparams.n_threads_batch = init_options.n_threads_batch;
    if (init_options.offload_kqv >= 0) cparams.offload_kqv = init_options.offload_kqv;
    if (init_options.flash_attn >= 0) cparams.flash_attn_type = init_options.flash_attn ?
        LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    state->context_params = cparams;
    
    state->ctx = llama_init_from_model(state->model, cparams);
    
    if (!state->ctx) {
        LlamaStateFree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create context", -1));
        return TCL_ERROR;
    }
    
    state->vocab = llama_model_get_vocab(state->model);
    if (init_options.n_seq_max > 1) {
        try { state->sequence_positions = new std::vector<int>(init_options.n_seq_max, -1); }
        catch (const std::exception &) { state->sequence_positions = NULL; }
        if (!state->sequence_positions) {
            LlamaStateFree((char*)state);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Memory allocation failed", -1)); return TCL_ERROR;
        }
        (*state->sequence_positions)[0] = 0;
        if (init_options.sequence_sampling) {
            try { state->sequence_sampling = new std::vector<SequenceSampling *>(init_options.n_seq_max, NULL); }
            catch (const std::exception &) {
                LlamaStateFree((char*)state);
                Tcl_SetObjResult(interp, Tcl_NewStringObj("Memory allocation failed", -1)); return TCL_ERROR;
            }
            state->sequence_generation = true;
        }
    }
    SamplingOptions defaults = SamplingOptionsFromState(state);
    state->sampler = CreateSampler(state, defaults);
    if (!state->sampler) {
        LlamaStateFree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create sampler", -1));
        return TCL_ERROR;
    }
    
    char handle[64];
    for (;;) {
        unsigned long long id = next_handle_id.load();
        if (id == std::numeric_limits<unsigned long long>::max()) {
            LlamaStateFree((char*)state);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Handle identity space exhausted", -1));
            return TCL_ERROR;
        }
        if (!next_handle_id.compare_exchange_weak(id, id + 1)) continue;
        snprintf(handle, sizeof(handle), "::llama%llu", id);
        if (!Tcl_FindCommand(interp, handle, NULL, TCL_GLOBAL_ONLY)) break;
    }
    state->command = Tcl_CreateObjCommand(interp, handle, LlamaHandleCmd, state, LlamaHandleDelete);
    if (!state->command) {
        LlamaStateFree((char*)state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create handle", -1));
        return TCL_ERROR;
    }
    
    Tcl_SetObjResult(interp, Tcl_NewStringObj(handle, -1));
    return TCL_OK;
}

static int Llama_Free_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Usage: llama::free handle", -1));
        return TCL_ERROR;
    }
    
    LlamaState *state = GetLlamaState(interp, objv[1], true);
    if (!state) return TCL_ERROR;
    Tcl_DeleteCommandFromToken(interp, state->command);
    return TCL_OK;
}

/* Report registered devices without loading a model or initializing a device. */
static int Llama_Devices_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 1) { Tcl_WrongNumArgs(interp, 1, objv, ""); return TCL_ERROR; }
    Tcl_Obj *devices = Tcl_NewListObj(0, NULL);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        struct ggml_backend_dev_props p = {};
        ggml_backend_dev_get_props(device, &p);
        const char *type = "unknown";
        switch (p.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: type = "cpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU: type = "gpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: type = "igpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accelerator"; break;
            case GGML_BACKEND_DEVICE_TYPE_META: type = "meta"; break;
        }
        Tcl_Obj *entry = Tcl_NewDictObj();
        Tcl_Obj *caps = Tcl_NewDictObj();
#define DEVICE_PUT(obj, key, val) Tcl_DictObjPut(interp, obj, Tcl_NewStringObj(key, -1), val)
        DEVICE_PUT(entry, "index", Tcl_NewWideIntObj((Tcl_WideInt)i));
        DEVICE_PUT(entry, "type", Tcl_NewStringObj(type, -1));
        if (p.name) DEVICE_PUT(entry, "name", Tcl_NewStringObj(p.name, -1));
        if (p.description) DEVICE_PUT(entry, "description", Tcl_NewStringObj(p.description, -1));
        if (p.device_id) DEVICE_PUT(entry, "device_id", Tcl_NewStringObj(p.device_id, -1));
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        const char *backend = reg ? ggml_backend_reg_name(reg) : NULL;
        if (backend) DEVICE_PUT(entry, "backend", Tcl_NewStringObj(backend, -1));
        if (p.memory_free <= (size_t)std::numeric_limits<Tcl_WideInt>::max())
            DEVICE_PUT(entry, "memory_free", Tcl_NewWideIntObj((Tcl_WideInt)p.memory_free));
        if (p.memory_total <= (size_t)std::numeric_limits<Tcl_WideInt>::max())
            DEVICE_PUT(entry, "memory_total", Tcl_NewWideIntObj((Tcl_WideInt)p.memory_total));
        DEVICE_PUT(caps, "async", Tcl_NewBooleanObj(p.caps.async));
        DEVICE_PUT(caps, "host_buffer", Tcl_NewBooleanObj(p.caps.host_buffer));
        DEVICE_PUT(caps, "buffer_from_host_ptr", Tcl_NewBooleanObj(p.caps.buffer_from_host_ptr));
        DEVICE_PUT(caps, "events", Tcl_NewBooleanObj(p.caps.events));
        DEVICE_PUT(caps, "mmap_support", Tcl_NewBooleanObj(p.caps.mmap_support));
        DEVICE_PUT(entry, "capabilities", caps);
        Tcl_ListObjAppendElement(interp, devices, entry);
#undef DEVICE_PUT
    }
    Tcl_SetObjResult(interp, devices);
    return TCL_OK;
}

/* Expose upstream registry names, advertised feature strings and support queries. */
static int Llama_BackendInfo_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    (void)cd;
    if (objc != 1) { Tcl_WrongNumArgs(interp, 1, objv, ""); return TCL_ERROR; }
    Tcl_Obj *result = Tcl_NewDictObj();
    Tcl_Obj *backends = Tcl_NewListObj(0, NULL);
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) continue;
        Tcl_Obj *entry = Tcl_NewDictObj();
        const char *name = ggml_backend_reg_name(reg);
        if (name) Tcl_DictObjPut(interp, entry, Tcl_NewStringObj("name", -1), Tcl_NewStringObj(name, -1));
        Tcl_Obj *features = Tcl_NewDictObj();
        ggml_backend_get_features_t get_features = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features) {
            struct ggml_backend_feature *list = get_features(reg);
            if (list) for (struct ggml_backend_feature *f = list; f->name; ++f) {
                Tcl_DictObjPut(interp, features, Tcl_NewStringObj(f->name, -1), Tcl_NewStringObj(f->value ? f->value : "", -1));
            }
        }
        Tcl_DictObjPut(interp, entry, Tcl_NewStringObj("features", -1), features);
        Tcl_ListObjAppendElement(interp, backends, entry);
    }
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("backends", -1), backends);
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("mmap", -1), Tcl_NewBooleanObj(llama_supports_mmap()));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("mlock", -1), Tcl_NewBooleanObj(llama_supports_mlock()));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("gpu_offload", -1), Tcl_NewBooleanObj(llama_supports_gpu_offload()));
    Tcl_DictObjPut(interp, result, Tcl_NewStringObj("rpc", -1), Tcl_NewBooleanObj(llama_supports_rpc()));
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
}

/* Create a sequence-local configuration from immutable handle defaults, never another sequence. */
static SequenceSampling *SequenceSamplingState(LlamaState *state, int id) {
    SequenceSampling *&sampling = (*state->sequence_sampling)[id];
    if (!sampling) sampling = new SequenceSampling(SamplingOptionsFromState(state));
    return sampling;
}

/* Raw token generation is synchronous: no callbacks or text-level filtering. */
static int GenerateSequence(Tcl_Interp *interp, LlamaState *state, int id,
                            int objc, Tcl_Obj *const objv[]) {
    if (!state->sequence_generation) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Requires sequence_sampling true at initialization", -1)); return TCL_ERROR;
    }
    if ((objc - 4) % 2) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Expected option-value pairs", -1)); return TCL_ERROR; }
    SequenceSampling *s = SequenceSamplingState(state, id);
    SamplingOptions options = s->options;
    bool configure = false, saw_max = false;
    int maximum = options.n_predict > 0 ? options.n_predict : 1;
    for (int i = 4; i < objc; i += 2) {
        const char *key = Tcl_GetString(objv[i]);
        if (strcmp(key, "-options") == 0 && !configure) {
            if (s->started) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Sampling options are frozen after generation starts; clear_cache or recreate sequence", -1)); return TCL_ERROR; }
            if (ParseSamplingDict(interp, objv[i+1], llama_vocab_n_tokens(state->vocab), &options) != TCL_OK) return TCL_ERROR;
            configure = true;
        } else if (strcmp(key, "-max_tokens") == 0 && !saw_max) {
            if (ParseOptionInt(interp, key, objv[i+1], &maximum, 1, 4096) != TCL_OK) return TCL_ERROR;
            saw_max = true;
        } else { Tcl_SetObjResult(interp, Tcl_NewStringObj("Unknown or duplicate sequence generation option", -1)); return TCL_ERROR; }
    }
    if (configure && options.n_predict != s->options.n_predict) {
        if (saw_max) { Tcl_SetObjResult(interp, Tcl_NewStringObj("num_predict and -max_tokens cannot be combined", -1)); return TCL_ERROR; }
        maximum = options.n_predict > 0 ? options.n_predict : 1;
    }
    if (s->logits.empty()) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Sequence requires prefill before generation", -1)); return TCL_ERROR; }
    if (configure || !s->sampler) {
        llama_sampler *sampler = CreateSampler(state, options);
        if (!sampler) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid sequence sampler or grammar", -1)); return TCL_ERROR; }
        if (s->sampler) llama_sampler_free(s->sampler);
        s->sampler = sampler; std::swap(s->options, options);
    }
    LlamaOperation operation(state);
    Tcl_Obj *result = Tcl_NewListObj(0, NULL);
    Tcl_IncrRefCount(result);
    int &position = (*state->sequence_positions)[id];
    int vocabulary = llama_vocab_n_tokens(state->vocab);
    for (int i = 0; i < maximum && !s->finished && position < (int)llama_n_ctx_seq(state->ctx); ++i) {
        llama_token token;
        s->started = true;
        if (SampleFromLogits(interp, s->sampler, vocabulary, s->logits.data(), &token) != TCL_OK) {
            ResetInferenceState(state); Tcl_DecrRefCount(result); return TCL_ERROR;
        }
        if (llama_vocab_is_eog(state->vocab, token) ||
            (llama_vocab_get_attr(state->vocab, token) & LLAMA_TOKEN_ATTR_CONTROL)) {
            s->finished = true; break;
        }
        struct llama_batch batch = llama_batch_init(1, 0, 1);
        fill_batch(batch, token, position, true);
        batch.seq_id[0][0] = id;
        int status = llama_decode(state->ctx, batch);
        llama_batch_free(batch);
        if (status == 0) state->sequence_fork_pending = false;
        const float *logits = status == 0 ? llama_get_logits_ith(state->ctx, -1) : NULL;
        if (!logits) {
            ResetInferenceState(state); Tcl_DecrRefCount(result);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed; all sequences reset", -1)); return TCL_ERROR;
        }
        s->logits.assign(logits, logits + vocabulary);
        ++position;
        if (id == 0) state->n_past = position;
        Tcl_ListObjAppendElement(interp, result, Tcl_NewIntObj(token));
    }
    Tcl_SetObjResult(interp, result); Tcl_DecrRefCount(result);
    return TCL_OK;
}

/* Dispatch sequence operations; exceptions never cross the Tcl command boundary. */
static int Llama_Sequence_Cmd(ClientData cd, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]) {
    try {
    (void)cd;
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "handle create|info|delete|prefill sequenceId ?tokenIds?"); return TCL_ERROR;
    }
    LlamaState *state = GetLlamaState(interp, objv[1]);
    if (!state) return TCL_ERROR;
    if (!state->sequence_positions) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Requires initialization with n_seq_max greater than 1", -1)); return TCL_ERROR;
    }
    const char *action = Tcl_GetString(objv[2]);
    bool prefill = strcmp(action, "prefill") == 0;
    bool generate = strcmp(action, "generate") == 0;
    bool fork = strcmp(action, "fork") == 0;
    if ((!prefill && !generate && !fork && objc != 4) || ((prefill || fork) && objc != 5) ||
        (!prefill && !generate && !fork && strcmp(action, "create") && strcmp(action, "info") && strcmp(action, "delete"))) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Expected create|info|delete id, or prefill id tokenIds", -1)); return TCL_ERROR;
    }
    int id;
    if (Tcl_GetIntFromObj(interp, objv[3], &id) != TCL_OK) return TCL_ERROR;
    if (id < 0 || id >= (int)state->sequence_positions->size()) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Sequence ID out of range", -1)); return TCL_ERROR;
    }
    int &position = (*state->sequence_positions)[id];
    llama_memory_t memory = llama_get_memory(state->ctx);
    if (!memory) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Model has no inference memory", -1)); return TCL_ERROR; }
    if (strcmp(action, "create") == 0) {
        if (position >= 0) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Sequence already registered", -1)); return TCL_ERROR; }
        position = 0;
        Tcl_SetObjResult(interp, Tcl_NewIntObj(id)); return TCL_OK;
    }
    if (position < 0) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Sequence is not registered", -1)); return TCL_ERROR; }
    if (fork) {
        if (state->sequence_fork_pending) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork copy pending; decode a token or clear_cache first", -1)); return TCL_ERROR;
        }
        int dest;
        if (Tcl_GetIntFromObj(interp, objv[4], &dest) != TCL_OK) return TCL_ERROR;
        if (dest <= 0 || dest >= (int)state->sequence_positions->size() || dest == id ||
            (*state->sequence_positions)[dest] >= 0) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork destination must be an unused nonzero sequence ID", -1)); return TCL_ERROR;
        }
        SequenceSampling *source = state->sequence_sampling ? (*state->sequence_sampling)[id] : NULL;
        if (position == 0 || (source && source->started)) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork requires prefill and no sampling", -1)); return TCL_ERROR;
        }
        char architecture[64];
        int archlen = llama_model_meta_val_str(state->model, "general.architecture", architecture, sizeof(architecture));
        if (archlen <= 0 || archlen >= (int)sizeof(architecture) ||
            (strcmp(architecture, "llama") && strcmp(architecture, "qwen3"))) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork is not supported for this model architecture", -1)); return TCL_ERROR;
        }
        if (llama_model_is_recurrent(state->model) || llama_model_is_hybrid(state->model) ||
            llama_model_n_swa(state->model) != 0) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork does not support recurrent, hybrid or sliding-window memory", -1)); return TCL_ERROR;
        }
        int min = llama_memory_seq_pos_min(memory, id), max = llama_memory_seq_pos_max(memory, id);
        if (min != 0 || max != position - 1) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork requires contiguous memory from position zero", -1)); return TCL_ERROR;
        }
        std::unique_ptr<SequenceSampling> copy;
        if (source) {
            copy.reset(new SequenceSampling(source->options));
            copy->logits = source->logits;
        }
        LlamaOperation operation(state);
        try {
            llama_memory_seq_cp(memory, id, dest, -1, -1);
        } catch (...) {
            Tcl_DeleteCommandFromToken(interp, state->command);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Native fork failed; handle closed", -1)); return TCL_ERROR;
        }
        if (llama_memory_seq_pos_min(memory, dest) != min || llama_memory_seq_pos_max(memory, dest) != max) {
            ResetInferenceState(state);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork could not be verified; all sequences reset", -1)); return TCL_ERROR;
        }
        (*state->sequence_positions)[dest] = position;
        state->sequence_fork_pending = true;
        if (state->sequence_sampling) (*state->sequence_sampling)[dest] = copy.release();
        Tcl_SetObjResult(interp, Tcl_NewIntObj(dest)); return TCL_OK;
    }
    if (generate) return GenerateSequence(interp, state, id, objc, objv);
    if (strcmp(action, "info") == 0) {
        Tcl_Obj *result = Tcl_NewDictObj();
        SequenceSampling *s = state->sequence_sampling ? (*state->sequence_sampling)[id] : NULL;
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("sampling_started", -1), Tcl_NewBooleanObj(s && s->started));
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("finished", -1), Tcl_NewBooleanObj(s && s->finished));
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("sequence", -1), Tcl_NewIntObj(id));
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("n_past", -1), Tcl_NewIntObj(position));
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("capacity", -1), Tcl_NewIntObj(llama_n_ctx_seq(state->ctx)));
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("pos_min", -1), Tcl_NewIntObj(llama_memory_seq_pos_min(memory, id)));
        Tcl_DictObjPut(interp, result, Tcl_NewStringObj("pos_max", -1), Tcl_NewIntObj(llama_memory_seq_pos_max(memory, id)));
        Tcl_SetObjResult(interp, result); return TCL_OK;
    }
    LlamaOperation operation(state);
    if (strcmp(action, "delete") == 0) {
        if (state->sequence_fork_pending) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Fork copy pending; decode a token or clear_cache first", -1)); return TCL_ERROR;
        }
        if (id == 0) { Tcl_SetObjResult(interp, Tcl_NewStringObj("Sequence 0 cannot be deleted; use clear_cache", -1)); return TCL_ERROR; }
        if (!llama_memory_seq_rm(memory, id, -1, -1)) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Cannot remove sequence memory", -1)); return TCL_ERROR;
        }
        if (state->sequence_sampling) {
            delete (*state->sequence_sampling)[id]; (*state->sequence_sampling)[id] = NULL;
        }
        position = -1; return TCL_OK;
    }
    int count;
    Tcl_Obj **items;
    if (Tcl_ListObjGetElements(interp, objv[4], &count, &items) != TCL_OK) return TCL_ERROR;
    if (count <= 0 || (uint32_t)count > llama_n_batch(state->ctx) ||
        count >= (int)llama_n_ctx_seq(state->ctx) - position) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Token count exceeds batch or sequence capacity, or is empty", -1)); return TCL_ERROR;
    }
    std::vector<llama_token> tokens(count);
    for (int i = 0; i < count; ++i) {
        int token;
        if (Tcl_GetIntFromObj(interp, items[i], &token) != TCL_OK) return TCL_ERROR;
        if (token < 0 || token >= llama_vocab_n_tokens(state->vocab)) {
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid token ID", -1)); return TCL_ERROR;
        }
        tokens[i] = token;
    }
    SequenceSampling *sampling = state->sequence_sampling ? SequenceSamplingState(state, id) : NULL;
    if (sampling && sampling->started) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Prefill after generation requires clear_cache or sequence recreation", -1)); return TCL_ERROR;
    }
    if (sampling) sampling->logits.resize(llama_vocab_n_tokens(state->vocab));
    struct llama_batch batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) {
        fill_batch(batch, tokens[i], position + i, i == count - 1);
        batch.seq_id[i][0] = id;
    }
    int status = llama_decode(state->ctx, batch);
    llama_batch_free(batch);
    if (status == 0) state->sequence_fork_pending = false;
    if (status != 0) {
        ResetInferenceState(state);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Decode failed; all sequences reset", -1)); return TCL_ERROR;
    }
    if (sampling) {
        const float *logits = llama_get_logits_ith(state->ctx, -1);
        if (!logits) { ResetInferenceState(state); Tcl_SetObjResult(interp, Tcl_NewStringObj("Logits unavailable; all sequences reset", -1)); return TCL_ERROR; }
        std::copy(logits, logits + sampling->logits.size(), sampling->logits.begin());
    }
    position += count;
    if (id == 0) state->n_past = position;
    Tcl_SetObjResult(interp, Tcl_NewIntObj(position)); return TCL_OK;
    } catch (const std::exception &e) {
        Tcl_SetObjResult(interp, Tcl_ObjPrintf("Sequence operation failed: %s", e.what())); return TCL_ERROR;
    }
}

int Tclllama_Init(Tcl_Interp *interp) {
    if (Tcl_InitStubs(interp, "8.6", 0) == NULL) {
        return TCL_ERROR;
    }

    if (Tcl_FindNamespace(interp, "::llama", NULL, 0) == NULL &&
        Tcl_CreateNamespace(interp, "::llama", NULL, NULL) == NULL) {
        return TCL_ERROR;
    }
    llama_backend_init();
    
    Tcl_CreateObjCommand(interp, "llama::version", Llama_Version_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::devices", Llama_Devices_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::backend_info", Llama_BackendInfo_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::init", Llama_Init_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::free", Llama_Free_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::generate", Llama_Generate_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::prefill", Llama_Prefill_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::sequence", Llama_Sequence_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::chat", Llama_Chat, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::tokenize", Llama_Tokenize_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::detokenize", Llama_Detokenize_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::clear_cache", Llama_ClearCache_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::snapshot_save", Llama_SnapshotSave_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::snapshot_restore", Llama_SnapshotRestore_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::snapshot_discard", Llama_SnapshotDiscard_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::get_context", Llama_GetContext_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::memory_info", Llama_MemoryInfo_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::memory_truncate", Llama_MemoryTruncate_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::info", Llama_Info_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "llama::verbose", Llama_Verbose_Cmd, NULL, NULL);
    
    return Tcl_PkgProvide(interp, "tclllama", "7.5");
}

#ifdef __cplusplus
}
#endif
