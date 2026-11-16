# Characterization record

This document records observed behavior under named configurations. It is not a
claim of universal compatibility.

## Verified build and tests

The regression run for the consolidated functional code used the local build
recipe in [BUILDING.md](BUILDING.md) and the eleven Tcl suites in `tests/`.
With `TCLLLAMA_TEST_CHAT=1`, every suite completed without skips or failures:

| Model | Result |
| --- | ---: |
| Llama 3.2 1B Instruct, Q8_0, SHA-256 `74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` | 44 passed / 44; 0 skipped; 0 failed |
| Qwen3 1.7B, Q4_K_M, SHA-256 `3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6` | 44 passed / 44; 0 skipped; 0 failed |

The test set covers argument parsing, compatibility, state continuity, GBNF,
lifecycle, logit bias, sampler behavior, JSON Schema, snapshots, Typical/DRY/XTC
and streaming. Chat checks were enabled for these model files. Exact test cases
and constraints are in the `tests/*.test` files.

Model-free native checks passed for sampler history, logit bias, candidate
validation, sampling strategies, schema conversion and stream buffering.
Generated JSON was parsed independently in the schema tests, but those checks
are not a general JSON Schema conformance test.

## Test conditions

- Platform: Linux x86_64; CPU backend.
- CPU: AMD Phenom II X4 965.
- Tcl: 8.6.14; compiler: GCC 15.3.1.
- llama.cpp: `1719dbb0ee7bdaf56de083079d3613bb670160a5`.
- Tests use local GGUF blobs outside the Git repository. No model download is
  part of the regression procedure.
- The test runs are functional checks, not extensive performance benchmarks.

## Observed limitations

- CPU execution was tested; GPU execution was not.
- Two model families do not establish support for every GGUF architecture,
  tokenizer, chat template, backend or hardware target.
- Chat behavior is template-dependent. The regression checks only these tested
  model templates; models without a supported template use the implementation's
  documented fallback and need separate characterization.
- Snapshot restore temporarily creates a second context and also retains the
  snapshot buffer. Snapshot memory cost was not separately measured.
- A token/context limit or callback interruption can leave a partial GBNF or
  JSON response. A syntactically generated prefix is not guaranteed to be a
  complete schema-conforming result.
- The model manager in `library/ollama_registry.tcl` has source-level commands
  for download, list, delete, information and path lookup, but it is not
  covered by a dedicated registry test suite in this baseline.
- No sanitizer or Valgrind run is recorded here.

For baseline revisions and exact build configuration, see
[BASELINE.md](BASELINE.md). Earlier per-iteration evidence remains in private
`AUDITORIA_DOCUMENTAL_001.md`–`022.md` files outside this repository; those
reports are source material, not repository content.

## Partial memory operations

On the same CPU baseline, `tests/memory.test` passed 3/3 with each recorded
model. It checks empty and populated sequence bounds, invalid positions,
suffix truncation, continued generation, snapshot restoration, clearing and
reentrant busy-handle rejection. This establishes the behavior for these two
models; windowed, recurrent, hybrid and GPU memory implementations remain
uncharacterized. Native removal failure after temporary-context creation was
not artificially injected. Truncation requires a temporary serialized state and
second context; its peak memory cost was not measured.

## Backend discovery

On the recorded Linux x86_64 CPU build, `llama::devices` reported one CPU
entry and `llama::backend_info` reported the CPU backend, `gpu_offload 0`,
`rpc 0`, `mmap 1` and `mlock 1`. Its feature dictionary reported `SSE3`,
`LLAMAFILE`, `OPENMP` and `REPACK`. These are upstream registry reports for
this build and machine, not independently verified instruction execution or GPU
inference. Memory figures are backend-reported snapshots. `tests/devices.test`
validates the shape without requiring a GPU or a specific manufacturer.

## Token batch prefill

The single-sequence prefill path is exercised by `tests/batch.test` using local
GGUF models. It checks token IDs and batch size, contiguous positions in sequence
0, continuation, snapshot restoration, memory truncation, clearing and busy
handles. The API does not establish multi-sequence independence or validate a
GPU backend. Native decode failure and all model-specific memory behaviors are
not exhaustively fault-injected.

## Shared-context prefill sequences

`tests/sequence.test` covers two registered sequences in one context, independent
positions, prefill, deletion without changing the other sequence, ID reuse,
whole-context clear, invalid IDs/tokens/batch size/capacity, busy handles and
rejection of unsupported sampling/snapshot operations. These observations concern
the local CPU decoder models, not GPU, recurrent or hybrid memory support.
No native decode failure is injected; whole-context reset on error is an explicit
recovery policy. Physical prefix-sharing and general branch separation are not
claimed. Native `n_ctx_seq` and allocation rounding determine effective capacity.
The functional sequence tests configure two sequences; the validation limit of
64 is not a claim that a 64-sequence workload has been tested.
On the final CPU build, all 15 Tcl suites passed with Llama 3.2 1B and Qwen3
1.7B: 59 tests per model, no failures or skipped tests, including four new
sequence tests. Native sampler-history, logit-bias, sampling-strategies and
schema-adapter synthetic checks also passed.

## Independent sequence sampling

`tests/sequence_generation.test` exercises equal seeded sequences while
alternating calls, continued constrained sampling with DRY/penalties, independent
grammars and JSON Schema/logit bias, validation failure preservation, deletion,
whole-context reset, unsupported operations and explicit initialization mode.
Only raw-token generation is characterized; callers reconstruct UTF-8 text from
accumulated token IDs. Independent generation with mid-stream option changes,
text callbacks or general shared-prefix branches is not supported. The mode
stores one vocabulary-sized float logits array per prefilled sequence; this is
additional host memory, not an inference memory benchmark. GPU, recurrent/hybrid
memories and exhaustive native failure injection remain unverified.

On the final functional build, all 15 previous Tcl suites passed for each local
model (59/59). The new sequence-generation suite passed its final five cases
with each model, including terminal-token isolation; it was repeated after that
last case was added. Thus 64 distinct cases across 16 suites were verified per
model without failures or skips. Native sampler-history, logit-bias,
sampling-strategies and schema-adapter synthetic checks also passed. Independent
Mirostat/XTC behavior is supported by the reused chain but not separately covered
by the new sequence tests; existing general sampling regressions remain relevant.

## Prefill fork — before Audit 035

`tests/fork.test` exercises native prefix copying and seeded continuation,
invalid IDs, delayed-copy restrictions and deletion of branches. The binding
verifies position metadata; this does not establish physical KV sharing or
memory savings. The tested CPU configurations use non-unified streams, where
llama.cpp schedules a buffer copy. Native failure injection, shared/unified
memory and other model families are not characterized.

With the final build at that stage, all 17 Tcl suites passed on each local CPU model: 70/70
tests, no skips or failures. The six fork tests include prefill-only continuation,
seeded generation, option/logit inheritance, deletion in both orders, invalid
requests and the busy-handle guard. No physical sharing or memory savings were
measured; recurrent, hybrid and sliding-window models are rejected rather than
considered verified.

## Fork after independent sampling — Audit 035

The fork path now permits a source in `independent` mode after sampling has
started, provided it is not finished and its stored logits and sampler are
consistent. The destination receives a copy of the current native KV prefix,
the source's cached next-token logits and options, all logical flags, and a
distinct sampler produced with `llama_sampler_clone`. This was exercised with
active GBNF, JSON Schema, Mirostat, logit bias, repetition penalty and DRY
configurations. A terminal sequence is rejected so the API cannot create a
branch whose deferred native copy has no subsequent generation step to
materialize it.

`tests/fork.test` checks matching positions immediately after the fork, one
decode on the source to materialize the copy without moving the destination,
individual continuation of both branches, `batch_step`, and deterministic
continuation after a multisequence snapshot restore. It also checks a second
fork from an active child and confirms sampler-clone isolation in the native
synthetic `sampler-history-test` by mutating one clone and observing that its
parent and peer clone retain their own penalty history. Each destination also
copies one vocabulary-sized float-logit array (`4 * n_vocab` bytes); sampler
clone memory depends on its components and was not measured.

Fork still uses the full-range KV copy and the model/memory restrictions from
the earlier characterization. The native function returns `void`; bounds
verification cannot compare copied KV bytes or prove physical buffer sharing.
Allocation/native-copy failure injection and GPU, recurrent/hybrid, sliding
window and non-`llama`/`qwen3` architectures remain unverified. The Tcl tests
run only on the local CPU backend and make no memory-savings claim.

On the final extension build, the complete 20-suite regression passed against
both local models with the active-fork tests included. Each run reports 97
total Tcl cases, 95 passed, two skipped by the existing `chat` constraint and
zero failures. The targeted `fork.test` suite passed 10/10 with each model
after its final JSON Schema test fix. The four native synthetic sampler,
schema and logit checks and the stream-buffer check also passed. These results
are limited to the local CPU backend.

## Joint multi-sequence prefill batches

`tests/batch_multisequence.test` exercises `llama::batch_prefill` with one native
decode call containing tokens assigned to different sequence IDs. The returned
`native_decode_calls` value is 1, while the test configures `n_ubatch` to 2 and
submits more than two total tokens, demonstrating that the request is not
implemented as a Tcl loop of individual prefill calls; any microbatching occurs
inside llama.cpp. The tests cover independent positions, logits retained for
each sampling sequence and subsequent alternating generation, prefill-only
mode, validation preservation, combined `n_batch` limits, pending fork copies
and busy-handle rejection.

The new suite passed 7/7 cases with Llama 3.2 1B Q8_0 and 7/7 with Qwen3 1.7B
Q4_K_M on the local CPU backend. This does not characterize GPU, AVX2 or AVX-512
execution. Native decode failure was not injected; on any nonzero status the
binding resets all sequences under its established recovery policy. Concurrent
generation and continuous batching remain unimplemented.

## Shared-round generation

`tests/batch_step.test` exercises `llama::batch_step` with independently
configured sequence samplers. The command clones each active sampler, samples
from its saved logits, places ordinary token IDs at their own sequence positions,
requests a logits row for each token, and invokes `llama_decode` once for the
joint token batch. `llama_get_logits_ith` maps those requested rows back to the
original batch indices; the binding copies them to the corresponding sequence
only after the native call succeeds. A terminal-only round performs no decode.

The tests cover two ordinary tokens in one call, repeated rounds, one finished
branch beside an active branch, grammar/JSON Schema/logit bias options, invalid
configuration preservation, batch limits, and a pending prefill fork. Tests
run with Llama 3.2 1B Q8_0 and Qwen3 1.7B Q4_K_M on the local CPU backend. They
do not establish CUDA, AVX2 or AVX-512 behavior, decode-failure atomicity, or
performance. llama.cpp may divide one public decode call into internal
`n_ubatch` work; this is not continuous batching or concurrent execution.
Sampler cloning temporarily increases host-side sampler memory during a round,
with no measured byte count.

## Whole-handle multisequence snapshots

`tests/snapshot_multisequence.test` exercises snapshots in both `prefill_only`
and `independent` modes. It verifies restored positions and registrations after
sequence creation/deletion, restoration after `clear_cache`, repeated restores,
rollback of a pending pre-sampling fork, and deterministic continuation of two
independent sampler/grammar states after a shared generation round. The existing
single-sequence snapshot suite remains applicable.

The native state is captured with llama.cpp's context-wide
`llama_state_get_size`/`llama_state_get_data` and restored using
`llama_state_set_data` in a temporary context. The binding separately copies
the per-sequence registry and positions, options, flags and saved logits, and
clones each mutable sampler. Position ranges are checked against the restored
native memory before the active context is swapped. A save while a fork copy is
pending is rejected; restoring a snapshot from before that fork replaces the
pending context and branch registry.

The implementation limits multisequence snapshots to `llama` and `qwen3`
decoder models that are neither recurrent nor hybrid and do not use
sliding-window attention. This guard is based on the model families available
and tested locally, not a universal guarantee for all GGUFs or backends. Restore
temporarily creates a second full context; peak memory was not measured. Native
failure injection, GPU backends, recurrent/hybrid memories and architectures
outside the tested family remain unverified. Snapshot data remains temporary,
in-memory and non-portable.

## Shared-round generation — final 033 regression

On the final source and test set, all 19 Tcl suites passed with both local CPU
models: 86/86 with Llama 3.2 1B Instruct Q8_0 and 86/86 with Qwen3 1.7B
Q4_K_M, with no skips. The four existing native synthetic checks
(sampler-history, logit-bias, sampling-strategies and schema-adapter) also
passed. The dedicated `batch_step` suite was rerun after its final additions
with 9/9 on each model. Its contexts use `n_ubatch 1` while advancing two
sequences, so their independently retained next logits are subsequently read
across llama.cpp's internal microbatch boundary.

A first test attempt exposed a native assertion during context creation when
`n_seq_max` exceeded `n_batch`: the selected revision reserves at least one
output per sequence but the configured output capacity is bounded by `n_batch`.
Initialization now rejects that combination in Tcl before loading the model or
creating the native context. The regression uses a nonexistent model path to
verify that option validation wins before model loading and cannot trigger that
native abort.

These results cover the local CPU build only. GPU, AVX2/AVX-512 and decode
failure injection were not tested. The result field `native_decode_calls` is
counted at the actual public `llama_decode` invocation; for two ordinary
selections it reports one. llama.cpp may internally process the request as
multiple `n_ubatch` microbatches.

## Whole-handle snapshots — Audit 034

The final audit build passed all 20 Tcl suites with both local CPU models:
93/93 tests for Llama 3.2 1B Instruct Q8_0 and 93/93 for Qwen3 1.7B Q4_K_M,
with no skips or failures. The new multisequence snapshot suite passed 7/7 on
each model. The four native synthetic checks (sampler history, logit bias,
sampling strategies and schema adapter) passed, as did the stream-buffer native
check.

The snapshot tests verify deterministic next-token continuation after restoring
independent sampler state, JSON Schema and GBNF progress, per-sequence logits,
DRY/repetition configuration, terminal state, sequence create/delete rollback,
repeated restores, `clear_cache` recovery and restoration from before a pending
fork. They do not inject allocation failures or native `llama_state_set_data`
errors, and they do not measure peak memory. The active context remains intact
until the temporary context and sequence resources have been prepared and the
native memory ranges validate. Runtime failure of a native call during process
execution is not proven recoverable beyond the explicit Tcl error paths.

Verification used the local Linux CPU build only. CUDA, AVX2, AVX-512, other
architectures, recurrent/hybrid memory, sliding-window models and snapshots
across llama.cpp versions or processes remain unverified or unsupported.
