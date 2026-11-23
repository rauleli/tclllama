# Decisions and open questions

This file records the current Tcl binding contracts separately from experiments.
The implementation source and [API.md](API.md) remain the operational authority.

## Current contracts

| Topic | Current contract | Basis |
| --- | --- | --- |
| Project and command identity | The project is named `tclllama`; its public Tcl commands are in the `llama::` namespace. Model operations use explicit opaque handles. | `generic/tclllama.c`, API and lifecycle tests. |
| Handle ownership | Context and model handles belong to their creating interpreter. A context owns its context, samplers, snapshots, sequence state and host logits, and retains a reference to its model owner. `llama::init` preserves the legacy one-model/one-context lifecycle through a private owner. Explicit model free fails while contexts retain it; deleting the model command drops only its explicit reference, and the native model is freed by the final reference release. Context deletion frees the context before releasing its model reference. Backend registration remains process-level. Foreign, stale, wrong-kind and cross-interpreter handles are rejected. | `LlamaModelOwner`, `LlamaStateFree`, lifecycle and shared-model tests; llama.cpp `llama_init_from_model`/`llama_free` API. |
| Direct GGUF inspection | `llama::gguf_open` owns a read-only `gguf_context` parsed from a file with the public gguf.h API and no tensor `ggml_context`; it is separate from model/context ownership. Scalar and bounded array values are copied into Tcl objects, and 64-bit unsigned values use exact decimal strings. | `GgufOwner`, `gguf_init_from_file` with `ctx = NULL`, public getter type checks and `tests/gguf.test`. |
| Shared-model execution | Multiple contexts may be created from one native `llama_model`; each has independent native context/KV and binding state. Inference using a shared model is serialized. A callback cannot reenter an operation on a sibling context while one context is active. This guard does not establish multithread or cross-interpreter safety. | Native pointer identity probe in the test build and shared-model integration tests. |
| Argument errors | Unknown, duplicate, malformed and out-of-range options return Tcl errors. Sampling configuration is parsed before it replaces the active configuration. | Parser implementation and `tests/arguments.test`. |
| Initialization | `llama::init` keeps its combined model/context option contract. The additive `model_load`/`context_create` API separates model keys (`n_gpu_layers`, `main_gpu`, `check_tensors`) from context keys (threads, batch, sequence count and context flags); invalid cross-layer keys are rejected. Context size defaults to 4096 and is accepted from 512 to 32768. GPU-related options are not evidence of GPU support. | Scoped `ParseInitOptions`, shared-model and argument tests. |
| Sampling persistence | Sampling dictionary values and `num_predict` persist on the handle. `-max_tokens` cannot be combined with `-options num_predict`. A reset restarts inference/sampler state while keeping configured values. | Parser, `CommitCallSampling`, continuity and sampler tests. |
| Generation and chat state | `generate` continues the existing context unless `-reset` is true. `chat` formats the supplied message list and resets inference state for each call. A later `generate` continues from the chat context. | `Llama_Generate_Cmd`, `Llama_Chat`, `tests/continuity.test`. |
| Failure recovery | Validation and sampler-preparation errors before native mutation preserve active inference state. Decode errors and failed post-copy verification reset the shared inference state when the handle remains live; an exception from the void KV-copy API closes the handle. Snapshot-restore failures before context replacement preserve the active context. Partial callback output already delivered cannot be recalled. | Code, `tests/failure_injection.test` (test-build hooks), fork integrity, argument, lifecycle and streaming tests. |
| Snapshots | Each handle owns at most one temporary in-memory snapshot. It captures native context and whole-handle logical state; independent multi-sequence mode also clones each sequence sampler and copies its logits, options and flags. Restore validates a temporary context before swapping. | Snapshot commands and `tests/snapshot.test`, `tests/snapshot_multisequence.test`. |
| Native llama.cpp state | `state_*` and `state_seq_*` expose the pinned engine's native byte-buffer and session-file APIs. Restores validate on a temporary context before replacement. Raw restore marks the handle native-state-only because llama.cpp state APIs do not restore tclllama samplers/options/sequence bookkeeping; inference resumes only after `clear_cache` or restoring a complete tclllama snapshot. `ON_DEVICE` is not exposed as a Tcl byte array. | `tests/native_state.test`, upstream `llama.h` and `llama-context.cpp`. |
| Independent sequence output | `sequence ... generate` keeps its raw token-list contract. `sequence ... generate_text` is the separate text result path and reuses that sequence's sampler/state; callbacks are synchronous, and callback failure follows whole-context reset recovery. | `tests/sequence_generation.test`, `tests/sequence_text.test` and Audit 038. |
| GBNF and JSON Schema | Grammar is persistent. JSON Schema uses llama.cpp's native converter through a deliberately restricted accepted subset. The modes are mutually exclusive. Early termination may return incomplete output. | API tests and source adapter. |
| Logit bias | Bias maps token IDs to finite adjustments or `suppress`. Grammar filtering precedes bias; bias cannot restore grammar-excluded tokens. Empty/invalid candidate sets return controlled errors in the guarded sampling path. | Native synthetic tests and model regressions. |
| Typical, DRY, XTC | The three native strategies default to disabled. Their combination with Mirostat 1/2 is rejected because the selected common Mirostat chain does not apply them. | `tests/strategies.test`, `tests/sampling_strategies.cpp`. |
| Dynamic temperature, Top-n-sigma, Adaptive-P | Dynamic temperature is opt-in (`temperature_delta > 0`) and otherwise preserves the existing temperature sampler. Top-n-sigma uses the common-chain position before Top-K and zero is a native no-op. Adaptive-P replaces the terminal distribution/greedy selector. In the pinned engine its sampler clone re-seeds RNG rather than copying advanced RNG state, so batch rounds, snapshots with active Adaptive-P and forks after it starts are rejected. | `tests/strategies.test`, native `sampling-strategies-test`, pinned `llama-sampler.cpp`. |
| Version labels | The current Tcl development package version is `0.1.0`; historical `7.5` is not current package metadata. `llama::version` describes the Tcl package, while `llama::engine_info` reports the pinned llama.cpp version. Public `1.0.0` is reserved for the joint tclllama and `tclllama-server` release. | TEA `PACKAGE_VERSION`, `Tcl_PkgProvide`, `llama::version`, `llama::engine_info`, Fase III plan. |

## Open decisions and limits

Device discovery uses the ggml registry without requiring a model. Upstream
feature strings and support flags are exposed as reported, without treating
them as evidence of successful inference or mapping enumeration indices to
the model's `main_gpu` setting.

The first partial memory operation truncates only a suffix of sequence 0.
It resets sampler history explicitly while preserving configured options; a
snapshot remains independent and may restore its original sampler. A temporary
context protects the active context from a failed native removal. Restricted
range deletion, copying, keep and position transforms are now exposed through
`sequence ... memory`; general automatic continuation across arbitrary edits
still requires further design.

- Whether persistent sampling settings should remain the long-term public contract or become per-call options. The present behavior is documented and retained.
- Native `llama.cpp` state APIs remain engine-format bindings and do not gain a tclllama wrapper format. After a compatible, contiguous, single-sequence restore on the supported conventional decoder memories, tclllama may resume by synchronizing the native position and constructing a fresh sampler from persistent options. Sequence restore resets only the destination sequence's sampler metadata and requires a new prefill to obtain logits. This is new sampling from restored KV, not exact restoration of sampler history, grammar progress, RNG/adaptive state or prior logits. Unsupported or ambiguous restores retain `native_state_only` and remain blocked from inference.
- Whether a future native-state interface should also serialize tclllama sampler and sequence metadata remains open; it would be a different contract from exposing llama.cpp's native state APIs and is not implemented here.
- Whether llama.cpp model sharing is safe for concurrent calls from separate Tcl threads or interpreters. The implementation intentionally serializes same-model operations in one interpreter and makes no broader concurrency claim.
- Whether prompt tokens should be accepted into repetition-penalty history. The current sampler history does not include the prompt; changing this affects output behavior.
- How future sampler rebuilds should replay prompt/generated history and adaptive or random sampler state. Rebuilding currently starts fresh sampler history.
- The `0.1.0` development version is not a public release. Final artifact naming, version migration policy and release packaging remain to be defined before `1.0.0`.
- Portable build/link strategy, including static versus shared llama.cpp libraries.
- GPU option behavior on actual GPU backends; no GPU hardware verification has been performed.
- Multi-thread safety and concurrent use of a handle. Interpreter ownership checks do not establish thread safety.
- Native GBNF edge cases that can hit llama.cpp assertions; the tested path includes guards, but no universal malformed-grammar safety guarantee is made.
- Broader JSON Schema support or general-purpose validation. The current converter integration accepts only the subset in API.md.
- Whether a token sampled as a stop should update sampler history, and broader semantics of prompt, sampled stop tokens and continuation.
- Broader model/template coverage, context-overflow behavior and memory tooling such as ASan/UBSan or Valgrind.

These questions do not by themselves block the current CPU-tested binding. Resolve them when a concrete feature or release requirement depends on them.

## Single-sequence token prefill

`llama::prefill` accepts token IDs and appends them as a single native batch in
sequence 0. Prompt tokens do not enter the sampler, matching `generate`; it
preserves sampler state and persistent options. The binding has one position
counter and one sampler per handle, so independent sequences and shared-prefix
branching require further state design. A failed native decode resets the active
inference state, while validation failures preserve it and snapshots remain usable.

A batch of tokens for one sequence, a batch mixing sequence IDs, independent
sequences within one context, and separate handles with separate contexts are
four distinct capabilities. The native API supports sequence IDs and memory
copying, but their availability does not by itself provide separate sampler
history or independent generation in this binding. Native `n_seq_max` and
context memory capacity must also be considered in a future design.

## Explicit multi-sequence prefill mode

`n_seq_max > 1` opts into a separate, prefill-only handle mode. Context/model/KV
memory are shared; a handle-owned table records registration (-1) or the next
position for every sequence. Sequence 0 is always registered. Positions are
updated only after successful native decode. There is no mutable sampler state
per sequence because no sampling is permitted in this mode. The legacy sampler
is retained as a handle resource but cannot drive generation in this mode.

This separation preserves default-handle contracts and prevents accidental
sharing of grammar, RNG and penalty history. Independent generation requires
sequence-owned samplers/options and explicit selection of output logits; native
batching alone does not provide those contracts. Multi-sequence snapshots are
available for the supported decoder memories and include native KV plus the
complete registration, position and sampling tables.
Native decode errors clear the entire context and reset registered positions.
`clear_cache` has the same whole-context scope. Delete removes a whole sequence;
whole-prefix branching after sampling is available through fork. The separate
`sequence ... memory` interface exposes restricted KV edits with fresh affected
sampling, while general automatic reconciliation remains unexposed.

## Joint prefill batches across sequences

`llama::batch_prefill` accepts at least two distinct registered sequences and
submits all their token entries through one native `llama_decode` invocation.
The binding validates the entire input, per-sequence capacity and combined
`n_batch` limit before decoding. Duplicate sequence IDs and empty token lists
are rejected. It does not split the request into multiple public decode calls;
llama.cpp may internally divide the call into `n_ubatch` microbatches.

Positions and per-sequence logits are committed only after a successful native
decode. The last token of each supplied sequence requests logits. In independent
sampling mode, each resulting logits row is copied into that sequence's private
host buffer, preserving subsequent alternating generation. A sequence that has
started sampling cannot be prefilled. A native decode error clears all sequence
memory and logical positions, matching the existing recovery policy. Snapshots
also capture this mode's per-sequence logits and mutable samplers.

## Independent sequence sampling: first stage

`sequence_sampling true` explicitly opts into independent raw-token generation
with `n_seq_max > 1`. The previous prefill-only mode and all default single
handles retain their contracts. Each registered sequence lazily owns options,
a fresh native sampler and a host copy of its last output logits. Native decode
replaces context-wide output logits; preserving them by sequence is necessary
for safe alternation. `llama_sampler_clone` is not used to initialize a sibling,
because cloning mutable history would inherit grammar/penalty/RNG progress.
Handle defaults are copied as configuration and native samplers are built fresh.

Sampling options become immutable once a sequence starts sampling. A caller can
change them after clear/recreation, preserving the established grammar/history
until then. This limitation avoids defining unapproved history replay on live
reconfiguration. Per-call token limits do not rebuild the sampler. Prefill after
sampling is similarly rejected in this first stage. Prompts do not enter sampler
history, consistent with the existing binding.

Raw token IDs are returned after their native decode succeeds. EOG/control
selection marks only the selected sequence finished. Text processing, callbacks,
chat templates and user stop IDs are not part of this new operation. Errors from
sampling or decode reset the entire shared inference state; configuration errors
do not. Whole-context reset preserves each sequence's configuration and resets
its sampler/logits/terminal state. Delete releases only its sequence resources.
Snapshots capture native memory and all logical sequence states together; saves
are rejected while a KV fork copy is pending. Restore replaces the native
context and registry together, and can roll back a pending fork from an earlier
snapshot. No portability or support for other memory families is claimed. There
is no concurrent or continuous scheduler.

## Sequence fork

Fork copies the full, contiguous native prefix using the API range `-1, -1`.
In independent mode, the destination also receives a copy of the source options,
logits and flags, plus `llama_sampler_clone` when a sampler exists. This preserves
the active grammar, penalty/DRY history, adaptive sampler state and RNG without
sharing a mutable sampler. A finished sequence is rejected to avoid creating a
branch whose deferred KV copy cannot be materialized by generation. The command
continues to accept only `llama` and `qwen3` architectures and rejects recurrent,
hybrid and sliding-window memory. Native success is checked through destination
position metadata, not KV byte comparison. On failed verification all sequence
inference state is reset; an exception from the void native copy closes the
handle. Before the copy is invoked, validation or sampler clone failure leaves
the active state unchanged. Cross-stream copies remain pending until a
successful decode; saves, additional forks and deletes stay rejected meanwhile.
After materialization, snapshots capture both branches together.

## Shared-round generation

`llama::batch_step` advances selected independent sequences by at most one
sampled token each. Its per-sequence options dictionary reuses the existing
sampling parser and can configure a sequence only before sampling begins.
Existing samplers are cloned for provisional sampling; newly configured
samplers are built from that sequence's saved options. The binding commits the
sampler clones only after a successful joint decode (or after a terminal-only
round that needs no decode). This preserves RNG, grammar, penalty and adaptive
sampler history without sharing mutable sampler state. A failed sampler choice
leaves the handle unchanged. A native decode error clears the complete shared
context and sequence state, following the established partial-microbatch
recovery rule.

This operation is a synchronous round primitive, not request scheduling,
continuous batching, parallel CPU execution or asynchronous Tcl processing.
Snapshots clone and restore all sequence samplers and logits for supported
decoder contexts.

## Adapter ownership and context replacement

LoRA commands own a native adapter and retain its model; attached contexts add
references. Command deletion may defer native destruction. Explicit release is
rejected while dependents exist. Adapter handles are confined to their Tcl
interpreter. Each context retains its own adapter set and control-vector data.
Changes require empty memory and no snapshot/pending fork. They are prepared on
a temporary context to preserve the active context on preparation failure.
All replacement-context paths reapply these context settings. State buffers
remain upstream native data, without a proprietary configuration wrapper.
Snapshots retain the same configuration because configuration changes are
blocked while a snapshot exists. No aLoRA invocation scheduler is introduced.

## Native model-loading arrays

Device references are process-local enumeration indices. Tensor weights require
an explicit ordered selection and are passed to llama.cpp without a binding
normalization algorithm. A model owner retains the native arrays through the
last shared context, even where the pinned engine copies individual fields.
Typed overrides use native structs plus an empty-key terminator. Omitted
n_gpu_layers now preserves the engine's -1 default; explicit zero retains CPU
weight loading. Progress callbacks and backend buffer-pattern overrides remain
unexposed until their Tcl thread/lifetime contracts are established.

## Explicit sequence-memory edits

`sequence ... memory` separates KV manipulation from logical fork. Conventional
CPU-tested decoder edits use native state cloning for protected replacement,
retaining model, LoRA and control-vector configuration. The wrapper prepares
fresh samplers only for affected sequences; unchanged sampling state is retained.
Known contiguous results synchronize counters, while removals creating holes or
nonzero minima leave KV intact in native_state_only. No interval database or
prompt reconstruction is added. Copy/shift work pending in the engine blocks
serialization and further edits until decode, including on traditional handles.
Snapshots from before editing remain a recovery option. Position transforms
are restricted to whole sequences with at most one populated sequence because
native cells may be shared; additional transforms require separate evidence.

## Context configuration and shared-model YaRN

Creation scalars are retained in each handle's native context_params; temporary
contexts reuse that recipe. `info context requested` is distinct from effective
public getters, because resolved RoPE/cache/output parameters lack universal
getters. Attention modes/pooling that require a different inference contract
remain rejected on generation contexts; runtime attention is query-only.
The pinned YaRN context factory writes model->hparams.n_ctx_train for a scale
unequal to training scale. The wrapper rejects that combination before context
creation (also for unspecified scaling with an explicit unequal scale); explicit
none/linear scaling is independent. Raw native state files cannot validate every
RoPE parameter, so compatible creation settings remain an import precondition.

## Cooperative control at decoded-token boundaries

Generation control is an explicit call-local Tcl command prefix polled only
before sampling, after any prompt decode. A false boolean is normal
cancellation; stream callback return values retain their previous meaning.
Cancellation does not accept a speculative next token, invalidate KV or finish
a sequence. Optional result dictionaries expose normal termination causes;
errors retain Tcl error semantics and existing shared-context recovery.

The native abort callback is intentionally not bound to Tcl scripts. In the
pinned CPU graph runner it can run on worker zero between graph nodes, outside
an interpreter-safe call site. Native decode status 2 leaves processed
microbatches in KV (as can fatal errors); it cannot be treated as a reversible
cooperative stop. No worker/event-loop/atomic cross-thread cancellation mechanism
is introduced in this synchronous binding. Such native cancellation remains a
separate contract, not an alias for the new pre-sampling callback.

## Token vocabulary query contract

`llama::vocab` returns copied, read-only Tcl values from the public native
vocabulary API. The stored vocabulary spelling is exposed as `text` and remains
distinct from a detokenized token piece. Missing special IDs keep native negative
sentinels; native suppression IDs are copied before return. Queries do not
change context or sampler state. API coverage levels and tool-calling placement
are recorded above and in the private TODO.

## Native detokenization result representation

The compatible two-argument `llama::detokenize` remains the existing
per-token display-piece operation. Native sequence detokenization is opt-in and
returns a Tcl ByteArray rather than assuming the native bytes are valid Tcl
UTF-8 text. The caller chooses any text decoding explicitly. Native token IDs
are range-checked before the C API is called, whose implementation asserts on
invalid vocabulary indices.

## Native text embeddings

Embedding inference uses a dedicated context kind over the existing shared
`llama_model` ownership. It enables native embeddings only for that context,
retains the model through the regular context reference, and rejects generation,
sampler, KV-edit, snapshot and state-transfer operations on the embedding handle.
This keeps the causal generation context contract and its pooling/attention
restrictions unchanged.

The initial Tcl operation handles one text per call, delegates sequence pooling
to llama.cpp, copies the returned `n_embd_out` vector, and clears native memory
before and after evaluation. `llama_encode` is used for encoder/non-causal
architectures and `llama_decode` for causal decoder contexts. Encoder-decoder
contexts, unpooled per-token outputs and RANK scores are not folded into the
single-vector contract. Task prefixes are caller input; no model-specific text
rewriter or vector store is introduced.

The follow-up batch operation assigns each input its own native sequence and
uses sequence pooling before copying results to Tcl. It is bounded by the
context's configured sequence and batch capacities; it does not automatically
split oversized input collections. Float32 byte output is explicitly
little-endian, with no header, so its byte length is four times the vector
dimension. The numeric-list and single-text APIs retain their existing
contracts. Runtime benchmarks remain host-specific characterization rather
than an API guarantee.
