# Decisions and open questions

This file records the current Tcl binding contracts separately from experiments.
The implementation source and [API.md](API.md) remain the operational authority.

## Current contracts

| Topic | Current contract | Basis |
| --- | --- | --- |
| Project and command identity | The project is named `tclllama`; its public Tcl commands are in the `llama::` namespace. Model operations use explicit opaque handles. | `generic/tclllama.c`, API and lifecycle tests. |
| Handle ownership | A handle belongs to the creating interpreter. Its delete callback schedules destruction; the state deleter releases snapshots and sequence samplers/logits, then the handle sampler, context and model. Explicit free, command deletion/rename-to-empty and interpreter teardown follow that path. Backend registration is process-level, not released per handle. Foreign, stale and cross-interpreter handles are rejected. | `LlamaHandleDelete`, `LlamaStateFree`, `tests/lifecycle.test`, `tests/lifecycle_resources.tcl` and Audit 037. |
| Argument errors | Unknown, duplicate, malformed and out-of-range options return Tcl errors. Sampling configuration is parsed before it replaces the active configuration. | Parser implementation and `tests/arguments.test`. |
| Initialization | The context size is positional, defaults to 4096 and is accepted from 512 to 32768. Model and context options are accepted through `-options`; GPU-related options are not evidence of GPU support. | `ParseInitOptions`, `Llama_Init_Cmd`, `tests/arguments.test`. |
| Sampling persistence | Sampling dictionary values and `num_predict` persist on the handle. `-max_tokens` cannot be combined with `-options num_predict`. A reset restarts inference/sampler state while keeping configured values. | Parser, `CommitCallSampling`, continuity and sampler tests. |
| Generation and chat state | `generate` continues the existing context unless `-reset` is true. `chat` formats the supplied message list and resets inference state for each call. A later `generate` continues from the chat context. | `Llama_Generate_Cmd`, `Llama_Chat`, `tests/continuity.test`. |
| Failure recovery | Validation and sampler-preparation errors before native mutation preserve active inference state. Decode errors and failed post-copy verification reset the shared inference state when the handle remains live; an exception from the void KV-copy API closes the handle. Snapshot-restore failures before context replacement preserve the active context. Partial callback output already delivered cannot be recalled. | Code, `tests/failure_injection.test` (test-build hooks), fork integrity, argument, lifecycle and streaming tests. |
| Snapshots | Each handle owns at most one temporary in-memory snapshot. It captures native context and whole-handle logical state; independent multi-sequence mode also clones each sequence sampler and copies its logits, options and flags. Restore validates a temporary context before swapping. | Snapshot commands and `tests/snapshot.test`, `tests/snapshot_multisequence.test`. |
| GBNF and JSON Schema | Grammar is persistent. JSON Schema uses llama.cpp's native converter through a deliberately restricted accepted subset. The modes are mutually exclusive. Early termination may return incomplete output. | API tests and source adapter. |
| Logit bias | Bias maps token IDs to finite adjustments or `suppress`. Grammar filtering precedes bias; bias cannot restore grammar-excluded tokens. Empty/invalid candidate sets return controlled errors in the guarded sampling path. | Native synthetic tests and model regressions. |
| Typical, DRY, XTC | The three native strategies default to disabled. Their combination with Mirostat 1/2 is rejected because the selected common Mirostat chain does not apply them. | `tests/strategies.test`, `tests/sampling_strategies.cpp`. |
| Version labels | Runtime/package value `7.5` is the historical code/package progression. It is not the complete library's public release number. The planned consolidated public release is `1.0.0`. | `Tcl_PkgProvide`, `llama::version`, project release plan. |

## Open decisions and limits

Device discovery uses the ggml registry without requiring a model. Upstream
feature strings and support flags are exposed as reported, without treating
them as evidence of successful inference or mapping enumeration indices to
the model's `main_gpu` setting.

The first partial memory operation truncates only a suffix of sequence 0.
It resets sampler history explicitly while preserving configured options; a
snapshot remains independent and may restore its original sampler. A temporary
context protects the active context from a failed native removal. Arbitrary
deletion, shifting and multi-sequence operations require further state and
continuation design.

- Whether persistent sampling settings should remain the long-term public contract or become per-call options. The present behavior is documented and retained.
- Whether prompt tokens should be accepted into repetition-penalty history. The current sampler history does not include the prompt; changing this affects output behavior.
- How future sampler rebuilds should replay prompt/generated history and adaptive or random sampler state. Rebuilding currently starts fresh sampler history.
- The final package version, TEA metadata and release packaging strategy needed to move from internal `7.5` to public `1.0.0`.
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
arbitrary trimming, copying/branching after sampling remains unexposed.

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
