# Tcl-Llama API Reference

Complete API reference for the Tcl-Llama extension.

## Handle lifecycle (current implementation)

The historical examples below use an older command form. The current binding
registers namespaced commands and operates on explicit, opaque handles:

```tcl
set h [llama::init model.gguf 512]
set tokens [llama::tokenize $h "Hello"]
llama::free $h
```

`llama::init path ?n_ctx? ?-options init_dict?` creates a handle owned by its Tcl interpreter;
the optional context size is an integer from 512 to 32768. `llama::free $h`
deletes the handle command and releases its sampler, context and model.
Removing the command with `rename $h {}` or deleting the owning interpreter
also releases those resources. After deletion, the handle is invalid, and
passing it or an unrelated Tcl command returns `Invalid handle`. A renamed
handle remains usable under its new command name. Treat handle names as opaque
and do not assume their representation or transfer them between interpreters.

During `llama::generate` or `llama::chat`, other operations on the same handle
return `Handle is busy` from a reentrant callback. A callback may free or
delete that handle; the active operation then returns `Handle was deleted
during callback`, and destruction is deferred until it stops using the state.
Callback errors are propagated. These contracts do not establish general
thread safety or guarantee behavior for every Tcl extension interaction.

## Current argument and option contract

The parser implemented by this checkout accepts these forms:

```tcl
set h [llama::init model.gguf ?n_ctx? ?-options init_dict?]
llama::generate $h prompt ?-callback command? ?-options dict? \
    ?-reset boolean? ?-stop_ids list? ?-system text? ?-max_tokens integer?
llama::chat $h messages ?-callback command? ?-options dict? \
    ?-stop_ids list? ?-max_tokens integer?
```

`n_ctx` defaults to 4096 and must be an integer from 512 through 32768.
The sampling dictionary accepts `temperature` (0–2), `top_k` (1–1000),
`top_p` and `min_p` (0–1), `repeat_penalty` (>0 through 2), `repeat_last_n`
(0–2048), `frequency_penalty` and `presence_penalty` (-2–2), `num_predict`
(-1 or a positive integer), `mirostat` (0–2), `mirostat_tau` (0–10),
`mirostat_eta` (0–1), and `seed` (-1 or a nonnegative integer). Unknown keys,
duplicate dictionary keys, malformed values, and out-of-range values return a
Tcl error. The command switches are likewise checked; a switch may appear at
most once, and `-max_tokens` cannot be combined with `-options num_predict`.
Stop token IDs must be integers in the loaded model vocabulary range.

### JSON Schema structured generation

`json_schema` is a JSON string in the persistent `-options` dictionary for
generate/chat. The binding parses it and calls llama.cpp's native converter,
forcing GBNF output. Empty strings disable either restriction mode. Active
`grammar` and `json_schema` are mutually exclusive, including persisted values;
switch explicitly, e.g. `-options [dict create grammar {} json_schema $schema]`.
Malformed JSON reports `JSON syntax error`; unsupported/invalid schemas report
`JSON Schema conversion error`. Configuration errors preserve the active
context and sampler. Sampling/termination errors follow the GBNF contract.

The supported subset is deliberately narrower than the native converter:

- Every schema node must be an object with an explicit single string `type`:
  object, array, string, integer, number, boolean, or null.
- Objects require a `properties` object and explicit `additionalProperties:false`.
  Optional `required` is an array of unique names declared in properties.
- Arrays require a single `items` schema. Optional `minItems`/`maxItems` are
  integers from 0 to 64, with minimum no larger than maximum. Omission of
  maxItems leaves the native grammar unbounded; it does not insert a maximum.
- Primitive types allow either `enum` (1 to 64 values) or `const`, with values
  matching that type. They cannot be combined with structural constraints.
- String `title` and `description` are permitted annotations. Other keywords
  are rejected, including refs, composition, pattern/format, numeric bounds,
  string length, uniqueItems, and schema version declarations. Boolean schemas
  and union type arrays are not accepted. Nesting depth is limited to 16 and
  each properties map to 64 entries. These are binding acceptance limits.

The native converter, not this binding, builds the grammar. Accepted output
is governed by that grammar, not by a general JSON Schema validator. Object
property ordering and whitespace follow native conversion rules. The native
JSON parser's handling of duplicate JSON keys is not additionally validated.
Continuation, reset, sampler reconstruction, snapshots, UTF-8 and partial
output behave as for GBNF. Incomplete output after a token limit need not be
valid JSON. No external refs are fetched and no automatic schema repair occurs.

### GBNF grammar

The persistent sampling option `grammar` accepts a GBNF string whose start
rule is `root`. An empty string disables it:

```tcl
set reply [llama::generate $h "Answer" -reset 1 \
    -options [dict create grammar {root ::= "yes" | "no"}]]
```

The same option works with `llama::chat`. Invalid grammars return a Tcl error
before the new sampler is installed. The grammar is applied before other
sampling components; a lack of selectable candidates or an invalid selection
returns an error and clears the inference state. Tokens are accepted only after
selection is checked. Native grammar implementation assertions remain a limit;
no guarantee is made for every malformed grammar or model.

Grammar configuration persists on the handle. Without a reset the grammar
continues from its previous generated tokens; `-reset` and chat restart it.
Changing sampling parameters rebuilds the sampler and resets grammar progress;
changing only `num_predict` or `-max_tokens` does not. Snapshots preserve the
grammar and its progress along with the rest of the sampler. With a grammar,
textual stop marker filtering is disabled, while native control and EOG tokens
and `-stop_ids` remain in effect. Token/context limits and callback errors can
leave an incomplete grammatical prefix; callbacks may already have received
that prefix.

### Logit bias

`-options` accepts a persistent `logit_bias` dictionary mapping vocabulary
token IDs to finite logit adjustments or the literal `suppress`:

```tcl
set reply [llama::generate $h "Answer" -reset 1 \
    -options [dict create logit_bias [dict create 42 2.0 99 suppress]]]
```

IDs must be integers from zero through `n_vocab - 1`, each ID may occur only
once, and at most 4096 entries are accepted. Finite values are limited to
`-100` through `100`. The value `suppress` maps to negative infinity in the
native llama.cpp sampler; it removes that token from selection, unlike a large
finite negative adjustment. An empty dictionary clears the setting. Malformed
or out-of-range values return a Tcl error without replacing the active sampler.

Bias is applied after grammar filtering and before penalties, top-k/top-p/min-p,
temperature, and the selected final sampler. It therefore cannot re-enable a
token excluded by GBNF or JSON Schema. The same ordering is used with Mirostat;
the bias changes its input candidate logits, while Mirostat retains its native
adaptive state. If all candidates are suppressed, restricted sampling returns
a Tcl error and resets inference rather than passing an empty selection to a
native selector. Without grammar or bias, the existing native sampling path is
unchanged. With bias but without grammar, a validated candidate path is used to
detect an empty candidate set before accepting a token.

Bias settings persist on the handle, are reported under
`llama::info $h`'s `sampling logit_bias` dictionary, and are copied by snapshots
along with the sampler state. Snapshot restore reinstates both their values and
the sampler's adaptive/history state. Reset restarts the sampler while keeping
the configured bias. Selected EOS/control tokens and `-stop_ids` still stop
generation under the existing rules, although a bias can change how likely a
token is to be selected. Callback and partial-output policies are unchanged. A
synthetic native test checks the numeric effect on known logits; generated text
alone is not used to prove the adjustment.

### Typical, DRY, and XTC samplers

The `-options` dictionary also exposes three native llama.cpp samplers:

| Option | Default | Valid values / effect |
| --- | ---: | --- |
| `typical_p` | `1.0` | `0.0`–`1.0`; `1.0` disables locally typical sampling. |
| `dry_multiplier` | `0.0` | `0.0`–`100.0`; `0.0` disables DRY. |
| `dry_base` | `1.75` | `1.0`–`100.0`; DRY repetition growth base. |
| `dry_allowed_length` | `2` | Integer `0`–`2048`; repeated prefix length tolerated by DRY. |
| `dry_penalty_last_n` | `64` | Integer `0`–`2048`; DRY history window. Must be positive when `dry_multiplier` is positive. |
| `dry_sequence_breakers` | `{"\n" ":" "\"" "*"}` | Tcl list of at most 64 unique nonempty strings, each at most 40 UTF-8 bytes. `{}` removes all sequence breakers. |
| `xtc_probability` | `0.0` | `0.0`–`1.0`; `0.0` disables XTC. |
| `xtc_threshold` | `0.1` | `0.0`–`0.5`; XTC probability threshold. |

Typical and XTC use native `min_keep = 1`; the sampling seed is the handle's
existing `seed`. DRY uses llama.cpp's default sequence breakers above. Its
native initialization scans the vocabulary for each configured breaker, so
enabling/rebuilding DRY can take longer than the other sampler setups. Invalid
parameters and duplicate/oversized breakers return Tcl errors before changing
the active sampler. All settings persist and are exposed in the `sampling`
dictionary returned by `llama::info`.

The chain order is the existing order with DRY after penalties, Typical after
top-k and before top-p, and XTC after min-p and before temperature. These
positions follow llama.cpp's common sampling sequence. The strategies combine
with logit bias and GBNF/JSON Schema; grammar filtering remains first, so these
samplers cannot reintroduce grammar-excluded tokens. They also retain native
clone/reset behavior and are included in handle snapshots. When all three are
disabled, no new sampler component is added and the existing sampling path is
preserved.

llama.cpp's common Mirostat construction uses a separate chain that omits DRY,
Typical, and XTC. Therefore tclllama rejects a configuration with Mirostat 1 or
2 whenever any of these strategies is enabled, rather than silently ignoring
the requested strategy. Disable the strategy (for example, set `typical_p 1`,
`dry_multiplier 0`, or `xtc_probability 0`) before enabling Mirostat.

### `llama::info` model and context data

`llama::info $handle` preserves its existing top-level keys (`n_ctx`, `n_past`,
`n_ctx_used`, `n_ctx_available`, `n_batch`, `n_ubatch`, thread counts,
`model_desc`, `n_vocab`, `model_size`, `model_n_params`, `sampling` and
`telemetry`). The new `model` subdictionary contains GGUF/model properties:

```tcl
set info [llama::info $handle]
set architecture [dict get $info model architecture]
set training_context [dict get $info model n_ctx_train]
set effective_context [dict get $info n_ctx]
```

The subdictionary provides `file_type`, `n_ctx_train`, `n_embd`, `n_embd_inp`,
`n_embd_out`, `n_layer`, `n_head`, `n_head_kv`, `vocab_type`, encoder/decoder
and recurrent/hybrid/diffusion flags, BOS/EOS/EOT token IDs, and tokenizer
add-BOS/add-EOS/add-SEP flags. Scalar metadata strings such as `architecture`,
`name`, `description`, `basename`, `finetune`, `size_label`, `tokenizer_model`
and `tokenizer_pre` are included only if present in the model. `model_desc`,
vocabulary count, file size and parameter count remain at their existing
top-level keys to avoid duplicate fields. Negative special-token IDs are the
native API's sentinel for an unavailable token. Training context describes the
model metadata; top-level `n_ctx` is the effective runtime context.

`llama::init` also accepts `-options` before or after the optional positional
context size. Its dictionary keys are `n_threads` (1–256), `n_threads_batch`
(1–256), `n_batch` (1–2048; default 2048 to preserve current behavior),
`n_ubatch` (1–2048 and no larger than `n_batch`), `n_gpu_layers` (-1–999;
default 0), `main_gpu` (0–10; default 0), and boolean `check_tensors`,
`offload_kqv`, and `flash_attn`. The thread counts and boolean context keys
retain llama.cpp defaults unless specified. `n_ctx` remains positional and
defaults to 4096. Model options (`n_gpu_layers`, `main_gpu`, `check_tensors`)
are applied before context creation; the remaining keys configure the context.
GPU configuration has not been verified on GPU hardware. Sampling options
passed to `-options` and `-max_tokens` are applied to
the handle and persist for later calls, matching the current implementation.
Whether these should instead be per-call settings remains an open API decision.
The older initialization table below contains keys that are not accepted by
the current command.

## Inference continuity (current implementation)

### Temporary snapshots

`llama::snapshot_save $h` replaces the handle's single private in-memory
snapshot and returns the native context state size in bytes. `llama::snapshot_restore $h`
restores it; `llama::snapshot_discard $h` releases it. Restore and discard
return an error if none exists. These operations reject a busy or invalid handle.
Snapshots are freed with the handle and cannot be transferred, saved to files,
or used in another process.

The snapshot includes native context memory, `n_past`, a cloned sampler and
the handle's persistent sampling parameters, including `num_predict`. Restore
reverts those parameters and clears the binding's telemetry. It creates a
temporary context with the same model and creation parameters, reads the native
state there, and swaps it in only after success. This temporarily requires
additional context memory, besides the snapshot buffer. Snapshots do not change
the model or initialization options. State portability across different models,
contexts, backends or llama.cpp versions is not provided. Exact continuation is
not guaranteed for every model, backend or source of nondeterminism.

`generate` appends prompt and generated tokens to the current context. Its
`-reset true` clears context and sampler history first; `clear_cache` does
the same. Each `chat` call clears that history and evaluates the supplied
message list using the model template. A subsequent `generate` continues from
the context left by chat. Callers must supply the desired conversation history
to each chat call. Chat does not accept `-reset`.

Configuration values remain on the handle across resets. Supplying changed
sampling parameters rebuilds the sampler and persists the resulting values;
changing only `-max_tokens` or `num_predict`, or repeating identical sampling
parameters, preserves its history. Sampled stop tokens may
enter sampler history without entering the context. The exact relationship
between sampler history, prompt tokens and per-call options remains an API
design question. An error from a streaming callback or a decode failure clears
the inference context and sampler history so the handle can be reused. Input
validation errors preserve the prior context. Callback errors can occur after
partial text has already been delivered to the callback.

The normal sampling chain applies penalties, top-k, top-p, min-p and
temperature, followed by a distribution selector (greedy when temperature is
zero). Mirostat modes instead use penalties, temperature and the selected
Mirostat selector; top-k, top-p and min-p do not apply in those modes. The
prompt is not currently fed to the sampler's penalty history. Rebuilding after
a change to a sampling parameter discards that history. Sampled stopping tokens
are accepted by llama.cpp before the binding recognizes them as stops.

The llama.cpp completion example feeds prompt tokens into its common sampling
chain. This binding currently does not: choosing whether repetition penalties
should also apply to the prompt changes output behavior and remains undecided.
The native penalty sampler can be cloned with its history intact, but changing
penalty parameters, the history window, the random seed, or Mirostat requires
separate decisions about replay and adaptive/random state. A new sampler starts
with empty history; changing sampling parameters currently has that effect.
The synthetic native regression in `tests/sampler_history.cpp` demonstrates
the difference between accepted, cloned, rebuilt, and reset penalty state.

## Streaming and termination

`-callback command` invokes the Tcl command synchronously with each available
text fragment. On success, concatenating the fragments in order produces the
returned string. Chunk sizes are not fixed. Callback errors interrupt generation,
propagate to Tcl, and clear inference state if the handle remains live; already
delivered fragments cannot be recalled. A callback may also delete the handle,
as described above. There is no separate cancellation or completion callback.

Generation stops at llama.cpp EOG or control tokens, a `-stop_ids` token, the
token or context limit, or a recognized textual stop marker. Stop tokens are
not emitted or decoded, though the sampler has already accepted them. Textual
markers are filtered across token boundaries. Valid UTF-8 sequences are kept
intact across chunks; malformed or incomplete sequences are replaced with U+FFFD.
An unfinished textual marker prefix at the end is returned as ordinary text.
The control-token rule is broader than EOG and may stop at other control tokens;
the fixed set of textual markers cannot cover every model convention.

## Module Loading

```tcl
package require tclllama
```

Or manually:

```tcl
load ./tclllama.so Tclllama
```

Once loaded, all `llama` commands become available.

## Command Reference

### Model Initialization and Cleanup

#### llama init

Initialize and load a model.

```tcl
llama init <model_path> ?options?
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | string | required | Path to GGUF format model file |
| options | dict | {} | Configuration options (see below) |

**Options Dictionary Keys:**

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| n_ctx | int | 128-32768 | 4096 | Context window size in tokens |
| n_batch | int | 32-2048 | 512 | Batch size for token processing |
| n_threads | int | 1-256 | system_count | Number of CPU threads |
| n_threads_batch | int | 1-256 | n_threads | Threads for batch processing |
| seed | int | -1, 0+ | -1 | Random seed (-1 for random) |
| rope_scaling_type | int | 0-2 | 0 | RoPE scaling method |
| rope_freq_base | float | 1.0-1000000 | 10000 | RoPE frequency base |
| rope_freq_scale | float | 0.001-1000 | 1.0 | RoPE frequency scale |
| yarn_ext_factor | float | 0.0-100 | 1.0 | YaRN extension factor |
| yarn_attn_factor | float | 0.0-10 | 1.0 | YaRN attention factor |
| yarn_beta_fast | float | 0.0-100 | 32.0 | YaRN beta fast |
| yarn_beta_slow | float | 1.0-10000 | 1.0 | YaRN beta slow |
| yarn_orig_ctx | int | 1-32768 | n_ctx | YaRN original context size |
| type_k | int | -1, 0-1 | -1 | KV cache type for keys |
| type_v | int | -1, 0-1 | -1 | KV cache type for values |
| gpu_layers | int | 0-999 | 0 | Layers to offload to GPU |
| main_gpu | int | 0-10 | 0 | Primary GPU index |
| tensor_split | string | "" | "" | GPU distribution for tensors |
| low_vram | bool | 0-1 | 0 | Enable low VRAM mode |
| mul_mat_q | bool | 0-1 | 1 | Use optimized matrix multiplication |
| logits_all | bool | 0-1 | 0 | Get logits for all tokens |
| embeddings | bool | 0-1 | 0 | Compute embeddings |
| offload_kqv | bool | 0-1 | 1 | Offload KQV to GPU |
| flash_attn | bool | 0-1 | 0 | Use flash attention |

**Returns:**
- TCL_OK on success
- TCL_ERROR on failure

**Example:**
```tcl
llama init "model.gguf" -n_ctx 2048 -n_threads 8
```

#### llama free

Unload the current model and free resources.

```tcl
llama free
```

**Parameters:** None

**Returns:**
- TCL_OK on success
- TCL_ERROR on failure (no model loaded)

**Example:**
```tcl
llama free
```

---

### Text Generation

#### llama generate

Generate text from a prompt.

```tcl
llama generate <prompt> ?options?
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| prompt | string | Input text to continue from |
| options | dict | Sampling and generation options |

**Generation Options:**

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| temperature | float | 0.0-2.0 | 0.8 | Token randomness |
| top_k | int | 1-1000 | 40 | Keep top-K most likely tokens |
| top_p | float | 0.0-1.0 | 0.95 | Cumulative probability threshold |
| min_p | float | 0.0-1.0 | 0.05 | Minimum probability for tokens |
| repeat_penalty | float | 0.0-2.0 | 1.1 | Penalty for token repetition |
| repeat_last_n | int | 0-2048 | 64 | Window for repeat penalty |
| frequency_penalty | float | -2.0-2.0 | 0.0 | Penalty based on token frequency |
| presence_penalty | float | -2.0-2.0 | 0.0 | Penalty for token presence |
| mirostat | int | 0-2 | 0 | Mirostat sampling (0=off, 1/2=on) |
| mirostat_tau | float | 0.0-10.0 | 5.0 | Mirostat target entropy |
| mirostat_eta | float | 0.0-1.0 | 0.1 | Mirostat learning rate |
| penalize_nl | bool | 0-1 | 1 | Penalize newline tokens |
| num_predict | int | -1, 1+ | -1 | Max tokens to generate (-1=infinite) |
| seed | int | -1, 0+ | -1 | Sampling random seed |

**Returns:**
- Generated text string

**Raises:**
- TCL_ERROR if no model is loaded
- TCL_ERROR if parameters are invalid

**Example:**
```tcl
# Simple generation
set response [llama generate "Hello, world" -num_predict 100]

# With sampling parameters
set response [llama generate "Explain AI" {
    temperature 0.7
    top_k 40
    top_p 0.9
    num_predict 256
}]

# Deterministic (temperature=0)
set response [llama generate "Q: What is 2+2?\nA:" {
    temperature 0.0
    num_predict 50
}]
```

**Temperature Effects:**
- `0.0` - Deterministic, always choose most likely token
- `0.5` - Low randomness, mostly predictable
- `1.0` - Balanced randomness
- `1.5` - High randomness, more creative
- `2.0` - Very high randomness, less coherent

**Token Limits:**
| Scenario | num_predict | Notes |
|----------|------------|-------|
| Short responses | 50-100 | Q&A, single sentences |
| Medium responses | 200-500 | Paragraphs, explanations |
| Long responses | 1000-2000 | Full articles |
| Unlimited | -1 | Use with caution |

---

### Tokenization

#### llama tokenize

Convert text to token IDs.

```tcl
llama tokenize <text> ?add_special?
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | string | required | Text to tokenize |
| add_special | bool | 1 | Include special tokens |

**Returns:**
- List of integer token IDs

**Example:**
```tcl
set tokens [llama tokenize "Hello, world!"]
puts $tokens  ;# Output: {1 Hello 29892 world 29991}

# Without special tokens
set tokens [llama tokenize "Hello" 0]
```

**Token ID Ranges:**
- 0-255 - Standard ASCII representation fallback
- 256+ - Model vocabulary tokens
- 65534 - Special token (often padding)
- 65535 - Unknown/out-of-vocabulary

#### llama detokenize

Convert token IDs back to text.

```tcl
llama detokenize <tokens> ?remove_special?
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tokens | list | required | List of integer token IDs |
| remove_special | bool | 0 | Remove special tokens from output |

**Returns:**
- Text string

**Example:**
```tcl
set text [llama detokenize {1 Hello 29892 world 29991}]
puts $text  ;# Output: " Hello, world!"

# Remove special tokens
set text [llama detokenize {1 Hello 29892} 1]
```

**Note:** Detokenized text may include whitespace padding from tokenization.

---

### Model Information

#### llama info

Get detailed information about the loaded model.

```tcl
llama info
```

**Parameters:** None

**Returns:**
Dictionary with keys:
| Key | Type | Description |
|-----|------|-------------|
| model_file | string | Path to loaded model file |
| context_size | int | Current context window size |
| vocabulary_size | int | Number of tokens in vocabulary |
| llama_version | string | llama.cpp version |
| architecture | string | Model architecture (e.g., "llama") |
| embedding_size | int | Size of embeddings |
| ffn_hidden_size | int | FFN hidden layer size |
| head_count | int | Number of attention heads |
| head_count_kv | int | Number of KV cache heads |
| layer_count | int | Number of transformer layers |
| rotary_dim | int | Rotary position embedding dimension |
| tokens_evaluated | int | Total tokens processed (input) |
| tokens_generated | int | Total tokens generated (output) |
| time_eval_ms | float | Time spent on token evaluation |
| time_gen_ms | float | Time spent on generation |

**Example:**
```tcl
set info [llama info]
puts "Model: [dict get $info model_file]"
puts "Context: [dict get $info context_size]"
puts "Vocab: [dict get $info vocabulary_size]"
puts "Layers: [dict get $info layer_count]"
puts "Tokens generated: [dict get $info tokens_generated]"
```

#### llama version

Get version and build information.

```tcl
llama version
```

**Parameters:** None

**Returns:**
Dictionary with keys:
| Key | Type | Description |
|-----|------|-------------|
| version | string | llama.cpp version string |
| build_commit | string | Git commit hash |
| build_number | int | Build number |

**Example:**
```tcl
set ver [llama version]
puts "llama.cpp [dict get $ver version]"
```

#### llama getcpuinfo

Get CPU information and detected features.

```tcl
llama getcpuinfo
```

**Parameters:** None

**Returns:**
Dictionary with keys:
| Key | Type | Description |
|-----|------|-------------|
| n_logical_cpus | int | Logical CPU count |
| n_physical_cpus | int | Physical CPU count |
| cpu_name | string | CPU model name |
| has_avx | bool | AVX support |
| has_avx2 | bool | AVX2 support |
| has_avx512 | bool | AVX-512 support |
| has_fma | bool | FMA support |
| has_sse3 | bool | SSE3 support |
| has_ssse3 | bool | SSSE3 support |
| has_sse4_1 | bool | SSE4.1 support |
| has_sse4_2 | bool | SSE4.2 support |

**Example:**
```tcl
set cpu [llama getcpuinfo]
puts "CPUs: [dict get $cpu n_logical_cpus]"
puts "AVX2: [dict get $cpu has_avx2]"
```

---

### Configuration

#### llama verbose

Set verbosity level for debug output.

```tcl
llama verbose <level>
```

**Parameters:**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| level | int | 0-3 | Verbosity level |

**Levels:**
- `0` - Silent (no output)
- `1` - Normal (errors only)
- `2` - Verbose (include info)
- `3` - Very verbose (include debug)

**Returns:**
- Previous verbosity level

**Example:**
```tcl
llama verbose 0  ;# Silent
llama generate "Test"

llama verbose 3  ;# Debug mode
llama generate "Test"
```

#### llama clearcache

Clear the KV (Key-Value) cache.

```tcl
llama clearcache
```

**Parameters:** None

**Returns:**
- TCL_OK

**Usage:**
Useful for:
- Freeing memory between inference runs
- Resetting context
- Improving performance with new prompts

**Example:**
```tcl
# After generation
llama generate "First prompt"

# Clear cache before new independent generation
llama clearcache

# New generation starts fresh
llama generate "Second prompt (no context from first)"
```

---

## Error Handling

All commands return TCL_OK on success or TCL_ERROR on failure.

**Common Errors:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "no model loaded" | No model initialized | Call `llama init` first |
| "model not found" | Model file missing | Check model path |
| "invalid parameter" | Bad option value | Check parameter ranges |
| "out of memory" | Insufficient RAM | Reduce context size |
| "failed to load" | Model corrupted | Verify model file |

**Example Error Handling:**
```tcl
if {[catch {llama init "model.gguf"} err]} {
    puts "Error loading model: $err"
    exit 1
}

if {[catch {llama generate "Test"} result]} {
    puts "Generation error: $result"
} else {
    puts "Result: $result"
}

llama free
```

---

## Performance Considerations

### Memory Usage

**Approximate Memory per Model:**
```
Memory ≈ Model Size + (Context Size × Hidden Size × 2)

For 7B model with 4K context:
≈ 7GB + (4K × 4096 × 2 bytes)
≈ 7GB + 32MB
≈ 7.03GB RAM
```

### Token Generation Speed

**Typical Performance (7B model, CPU):**
- First token: 1-5 seconds (prompt processing)
- Subsequent tokens: 0.5-2 seconds each (generation)
- Can vary significantly by CPU

**Optimization Tips:**
1. Reduce context size for faster initial processing
2. Use batch processing for multiple prompts
3. Reuse model across multiple generations
4. Adjust `top_k` to reduce candidate evaluation
5. Use `num_predict` to limit output

### Sampling Parameter Trade-offs

| Parameter | Speed Impact | Quality Impact |
|-----------|--------------|----------------|
| temperature | None | High |
| top_k | Medium | Medium |
| top_p | Low | Medium |
| repeat_penalty | Low | Low |
| mirostat | High | High |

---

## Examples

### Interactive Chat

```tcl
#!/usr/bin/tclsh

package require tclllama

# Load model
if {[catch {llama init "model.gguf" -n_ctx 2048} err]} {
    puts "Failed to load model: $err"
    exit 1
}

llama verbose 0
set history ""

while {1} {
    puts -nonewline "> "
    flush stdout

    if {[gets stdin line] < 0} break
    if {$line eq "exit"} break

    append history "User: $line\n"

    if {[catch {
        set response [llama generate $history -num_predict 256 -temperature 0.7]
    } err]} {
        puts "Error: $err"
        continue
    }

    puts "AI: $response\n"
    append history "AI: $response\n"
}

llama free
```

### Batch Processing

```tcl
#!/usr/bin/tclsh

package require tclllama
package require json

llama init "model.gguf" -n_ctx 1024 -n_threads 8

set input [open "prompts.jsonl" r]
set output [open "results.jsonl" w]

set count 0
while {[gets $input line] >= 0} {
    set obj [json::json2dict $line]
    set prompt [dict get $obj prompt]

    set result [llama generate $prompt -num_predict 200]

    dict set obj response $result
    puts $output [json::dict2json $obj]

    incr count
    if {$count % 10 == 0} {
        puts "Processed: $count"
        flush stdout
    }
}

close $input
close $output
llama free

puts "Complete: $count prompts processed"
```

### Model Comparison

```tcl
#!/usr/bin/tclsh

package require tclllama

set prompt "Explain quantum computing"
set models {"model1.gguf" "model2.gguf" "model3.gguf"}

set test_params {
    temperature 0.7
    top_k 40
    num_predict 200
}

foreach model $models {
    puts "\n=== $model ==="

    llama init $model -n_ctx 2048
    set info [llama info]
    puts "Params: [dict get $info layer_count] layers, [dict get $info vocabulary_size] vocab"

    set result [llama generate $prompt {*}$test_params]
    puts "Response: [string range $result 0 200]..."

    llama free
}
```

---

## Related Resources

- **README.md** - User guide and quick start
- **BUILDING.md** - Compilation instructions
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Model Hub**: https://huggingface.co/models?library=gguf

---

**Version**: 1.0
**Last Updated**: December 2024
**Compatibility**: Tcl 8.6+, llama.cpp latest
