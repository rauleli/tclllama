# Tcl-Llama API Reference

Current command and behavior reference for the `tclllama` Tcl extension.

## Handle lifecycle (current implementation)

### Backend and device discovery

`llama::devices` needs no model or handle. It returns a list of dictionaries
for devices in the ggml registry. Each includes an enumeration `index`, a
`type` (`cpu`, `gpu`, `igpu`, `accelerator`, `meta` or `unknown`), and a
`capabilities` dictionary with Tcl booleans `async`, `host_buffer`,
`buffer_from_host_ptr`, `events` and `mmap_support`. When reported, `name`,
`description`, `backend`, `device_id`, `memory_free` and `memory_total` are
included. Memory values are upstream-reported bytes at query time; zero or
equal free/total values need not indicate accurate available memory.

`llama::backend_info` returns `backends` (a list of registry names and their
upstream `features` string dictionaries) and upstream boolean support queries
`mmap`, `mlock`, `gpu_offload` and `rpc`. Feature names and values are supplied
by each backend; they can reflect build options or runtime detection and are
not a portable promise about a particular CPU instruction or inference path.
An empty feature dictionary may mean the backend does not expose the query.
Support queries may cause upstream to load available backends when its registry
is empty. Neither command runs inference or verifies a device.

Device enumeration indices are not a portable mapping to `main_gpu`, which is
passed to llama.cpp's model parameters and depends on its device selection and
split mode. `n_gpu_layers` requests offload; a detected GPU or a true support
flag does not prove that offload will run successfully. The initialization
options and defaults are unchanged.

The binding registers namespaced commands and operates on explicit, opaque
handles:

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

Defaults for the principal sampling options are:

| Key | Default | Accepted range |
| --- | ---: | --- |
| `temperature` | `0.8` | `0`–`2` |
| `top_k` | `40` | `1`–`1000` |
| `top_p` | `0.95` | `0`–`1` |
| `min_p` | `0.05` | `0`–`1` |
| `repeat_penalty` | `1.1` | greater than `0` through `2` |
| `repeat_last_n` | `64` | `0`–`2048` |
| `frequency_penalty`, `presence_penalty` | `0` | `-2`–`2` |
| `num_predict` | `-1` | `-1` or a positive 32-bit integer |
| `mirostat` | `0` | `0`–`2` |
| `mirostat_tau` | `5` | `0`–`10` |
| `mirostat_eta` | `0.1` | `0`–`1` |
| `seed` | `-1` | `-1` or a nonnegative 32-bit integer |

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
Initialization options and defaults are listed in the preceding paragraph;
unsupported legacy keys are rejected.

## Inference continuity (current implementation)

### Inspecting and truncating inference memory

`llama::memory_info $h` returns a dictionary with `sequence` (always 0),
`pos_min`, `pos_max`, `n_past` and `can_shift`. Native positions are inclusive;
an empty sequence reports -1 for both position bounds. `can_shift` reports a
native capability, not an exposed shift operation. Windowed caches can have a
nonzero minimum, so position bounds do not imply a physical KV cell count.

`llama::memory_truncate $h position` removes the suffix from `position`
inclusive onward and returns the new position. The integer must be between 0
and `n_past`. A position equal to `n_past` is a no-op. Partial truncation
requires native positions to cover the contiguous range from 0 through
`n_past - 1`; windowed or incompatible memory is rejected. Position 0 clears
the sequence. The native operation is attempted in a temporary context and
failure leaves the active context unchanged. On success the sampler is reset,
including grammar and random/penalty history, while persistent options remain;
telemetry is cleared. This intentionally does not reconstruct sampler history
for retained tokens. The next `generate` appends to the retained prefix.

The existing snapshot remains unchanged and can restore its saved context and
sampler. Memory operations reject busy and invalid handles. Temporary restore
requires additional memory. Other sequence IDs, holes, shifts and arbitrary
range deletion are not supported by this binding.

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


## Loading and command reference

Load the package with:

```tcl
package require tclllama 7.5
```

The package registers these public commands:

| Command | Purpose |
| --- | --- |
| `llama::devices` | List registered devices without loading a model. |
| `llama::backend_info` | Query registered backends and native support flags. |
| `llama::init model_path ?n_ctx? ?-options dict?` | Load a GGUF and create a Tcl handle. |
| `llama::free handle` | Delete a handle and release its native resources. |
| `llama::generate handle prompt ?switches?` | Continue or reset text generation. |
| `llama::chat handle messages ?switches?` | Format messages and generate a response. |
| `llama::tokenize handle text` | Return token IDs for text. |
| `llama::detokenize handle token_list` | Convert token IDs to text. |
| `llama::info handle` | Return model, context, sampling and telemetry dictionaries. |
| `llama::get_context handle` | Return the current `n_past` token position. |
| `llama::memory_info handle` | Inspect native sequence 0 position bounds and shift capability. |
| `llama::memory_truncate handle position` | Remove a sequence 0 suffix and reset sampler history. |
| `llama::prefill handle tokenIds` | Decode a batch of prompt tokens into sequence 0. |
| `llama::clear_cache handle` | Reset context, sampler history and telemetry. |
| `llama::snapshot_save handle` | Replace the handle's in-memory snapshot. |
| `llama::snapshot_restore handle` | Restore the saved snapshot. |
| `llama::snapshot_discard handle` | Discard the saved snapshot. |
| `llama::verbose handle ?boolean?` | Read or set per-handle diagnostic output. |
| `llama::version` | Return tclllama package/build metadata. |

A chat message is a Tcl dictionary with non-empty `role` and a `content` field.
The message list is formatted using the model's `tokenizer.chat_template` when
available; otherwise the implementation uses a simple role/content fallback.
Template availability and behavior depend on the GGUF. `llama::chat` resets
inference state on each call, so pass the conversation history required for
that turn. A later `llama::generate` continues from the context left by chat.

`llama::verbose` accepts Tcl booleans and returns the current boolean value; it
is not a multi-level logging control. `llama::version` returns `version`,
`edition` and `compiled`. Its `version` value is the current internal package
number `7.5`, not a llama.cpp version or the public release version. The
historical code progression is distinct from the planned **tclllama 1.0.0**
release. Resolving the package metadata for distribution remains open.

`llama::tokenize` and `llama::detokenize` require a loaded handle. Tokenization
uses the current model vocabulary and special-token behavior of the binding;
exact token IDs are model-specific. `llama::info` separates model metadata
(including training context) from effective context settings. Fields whose GGUF
metadata is absent are omitted where the native query can detect absence.

The command options and failure behavior described above are the current
implementation contract. Historical singleton-style command examples and
superseded option tables are intentionally not part of this reference.

### Token batch prefill

`llama::prefill handle tokenIds` decodes a nonempty Tcl list of vocabulary token
IDs as prompt input in sequence 0 and returns the resulting `n_past`. Positions
are assigned consecutively starting at the current `n_past`; the batch cannot
exceed the effective `n_batch`, and at least one context position must remain.
The caller supplies any desired beginning-of-sequence token. The command does
not generate text, sample tokens or accept prompt tokens into the sampler;
this follows the existing `generate` prompt policy. It preserves persistent
sampling options and the sampler's current state. `generate` can append another
prompt and continue afterward. `chat` still resets inference as documented.

The list and token IDs are validated before decoding. Invalid input leaves the
state untouched. A native decode failure clears the active inference memory and
resets the sampler, as in `generate`, since native decode can partially modify
memory. Snapshots remain available; `snapshot_restore`, `clear_cache` and
`memory_truncate` retain their existing contracts. A busy or invalid handle is
rejected. The native batch is allocated and freed within the call. This API
supports only sequence 0; it does not expose independent sequences or their
samplers.

### Shared-context prefill sequences

Initialization accepts `n_seq_max` (1–64, default 1). A value greater than 1
selects **prefill-only multi-sequence mode** unless `sequence_sampling` is enabled.
The default retains
all existing behavior. `llama::info` adds `n_seq_max`, `n_ctx_seq` (effective native
capacity per sequence) and `sequence_mode` (`single`, `prefill_only` or `independent`). Native
rounding can change the effective total context allocation. Existing `n_past`,
`n_ctx_used` remain sequence-0 diagnostics, not aggregate KV occupancy.
`n_ctx_available` retains its legacy calculation (requested `n_ctx` minus
sequence-0 position); in multi mode it is not a usable per-sequence capacity.
Use the sequence query for per-sequence position and capacity.

```tcl
set h [llama::init $model 512 -options {n_seq_max 2}]
llama::sequence $h create 1
set ids [llama::tokenize $h {Hello world}]
llama::sequence $h prefill 0 $ids
llama::sequence $h prefill 1 $ids
puts [llama::sequence $h info 1]
llama::sequence $h delete 1
llama::free $h
```

`llama::sequence handle action sequenceId ?tokenIds?` requires multi-sequence
mode. IDs range from 0 through `n_seq_max - 1`; sequence 0 is registered at init.

- `create id`: registers an empty sequence, returns its ID; duplicates are errors.
- `info id`: returns a dict with `sequence`, `n_past`, `capacity`, `pos_min` and
  `pos_max`. Native empty positions are -1. Unregistered IDs are errors.
- `prefill id tokenIds`: appends a nonempty list of vocabulary token IDs at
  contiguous positions in that sequence. Returns its new `n_past`. It must fit
  the effective `n_batch` and leave one position within `n_ctx_seq`. No text or
  sampling is performed. Callers supply BOS tokens if desired.
- `delete id`: removes all native memory belonging to the registered sequence
  and unregisters it; returns an empty result. ID 0 cannot be deleted.

Validation errors preserve all sequence positions and memory. Native decode
failure resets **all** sequences because partially processed microbatches may
remain; registrations survive with position zero. `clear_cache` likewise clears
all native memory and resets registered positions. Memory queries and
`get_context` refer to sequence 0. Every sequence operation rejects busy handles.
All logical sequence state is released with the owning handle.

In multi-sequence mode, `generate`, `chat`, legacy `prefill`, `memory_truncate`,
`snapshot_save` and `snapshot_restore` return an explicit error. The existing
snapshot format does not contain per-sequence state, and the existing sampler
must not be shared between independent sequences. Tokenization, model queries
and freeing remain available. In `prefill_only`, no per-sequence sampler is created or mutated;
that mode provides shared-context prefill, not independent text generation,
mixed-sequence batches, general prefix branching or asynchronous execution. Support
beyond the two tested decoder models remains model and memory dependent.

### Independent raw-token generation by sequence

Initialize with `-options {n_seq_max 2 sequence_sampling true}` to select
`sequence_mode independent`. `sequence_sampling` is a boolean, default false,
and requires `n_seq_max > 1`; the previous `prefill_only` mode remains available.
The new operation is:

```tcl
llama::sequence $h generate $id ?-options samplingDict? ?-max_tokens count?
```

It returns a **list of token IDs**, not a text string. Prefill the registered
sequence first. Each sequence owns its options, native sampler and a copy of its
last logits; alternating calls cannot use the other sequence's output logits.
New sequences inherit the immutable handle sampling defaults, never another
sequence's current options or sampler history. Native samplers are built fresh,
not cloned from a mutable sibling. Prompt tokens do not enter their histories.

The sampling dictionary uses the same validation and strategies as the existing
binding, including GBNF, JSON Schema, logit bias, DRY, Typical, XTC and Mirostat
with the existing combination restrictions. Configuration persists per sequence.
It may be set before the first sampling operation. After sampling starts,
`-options` is rejected; recreating the sequence or `clear_cache` allows new
configuration without silently discarding live grammar/RNG/history state.
`-max_tokens` is per call (1–4096); it does not rebuild the sampler. With no
explicit limit, a positive persistent `num_predict` is used, otherwise one token.
Changing `num_predict` together with `-max_tokens` is rejected.

Each ordinary token is decoded into its own sequence before being returned.
EOG/control tokens terminate the sequence, are accepted by its sampler but are
not decoded or returned. `info` adds booleans `sampling_started` and `finished`.
A finished sequence returns an empty token list on continuation. Capacity
exhaustion also returns an empty list. Partial output at the token limit need
not complete a grammar. Since this is a raw-token operation, textual markers are
not filtered and UTF-8 bytes are not independently decoded per token. Accumulate
the returned IDs and detokenize the full list to reconstruct text.

There is no `-callback`, `-stop_ids`, prompt argument or chat-template processing
on this operation. Prefill after sampling starts is rejected; use clear/recreate.
Legacy `llama::generate`/`llama::chat` remain unavailable on multi-sequence handles.
Snapshots and partial truncation remain unavailable in either multi mode.
`clear_cache` resets all positions, logits, grammar/history/RNG state and terminal
flags while retaining registrations and each sequence's persistent configuration.
Deleting a nonzero sequence frees its sampler/logits/options; recreation inherits
fresh defaults. Sequence 0 remains reserved and requires whole-context clear to
restart or change its frozen configuration. Sampling failure or native decode failure resets all sequences;
configuration validation/conversion errors preserve the previous active state.
This is synchronous independent sampling, not mixed-sequence batching,
continuous batching, concurrency or asynchronous inference.

### Forking a prefilled sequence

`llama::sequence $h fork $source $destination` registers a new destination
and copies the source's full native KV sequence. The source must be registered,
nonempty, contiguous from position zero and not yet have started sampling. The
destination must be an unused nonzero ID. Initially this operation is limited to
GGUF architectures `llama` and `qwen3`; other architectures return an error.
Recurrent, hybrid and sliding-window models are rejected explicitly.
It works in prefill-only mode and independent-sampling mode. In the latter it
copies the source's persistent options and cached logits, and constructs a fresh
sampler for each branch only when sampling starts. It does not clone a sampler
whose sampling has already started.

In the selected native implementation, non-unified streams schedule an actual
buffer copy for the next decode. Until that decode succeeds, another fork or
sequence deletion is rejected; `clear_cache` clears the context. After the
copy materializes, either branch can be deleted independently. The operation
verifies the native destination's position bounds, not the bytes in the KV
buffer. A failed verification resets all sequence states; if the native copy
throws an exception, the handle is closed. Allocation and validation errors
before the native call preserve the original state. No atomicity or physical
memory savings are promised. Both branches still share one native context.
