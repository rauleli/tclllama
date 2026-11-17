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

## KV fork integrity, fault paths and memory — Audit 036

### Functional KV continuity

`tests/fork_integrity.test` compares deterministic GBNF-constrained continuations
against no-fork references. It exercises a source continuation after fork, both
branches continued independently, alternating branch order, deleting the
destination, deleting a nonzero source after its fork is materialized, and
forking an already active child followed by `batch_step` and snapshot restore.
All three cases passed with Llama 3.2 1B Instruct Q8_0 and Qwen3 1.7B Q4_K_M on
the local CPU backend. The token IDs, sequence positions and deterministic
restored rounds matched their references. This is behavioral evidence that
the KV state is usable; it does not inspect or compare the native KV bytes and
does not demonstrate physical memory sharing.

Fork sources, destinations and branches retained their expected logical
positions in the tested sequences. Pending copies became usable after decode;
snapshots remained rejected until then. Existing and new tests exercised fork
cascades, branch removal, `clear_cache`, shared rounds and restoration after
materialization. No reproducible functional KV corruption was found, so no
production recovery policy or inference code was changed in this audit.

### Controlled failure paths

A separate extension was built with `-DTCLLLAMA_TESTING` in
`build/audit036-test`. Only that build registers the private
`::tclllama::test::fail_next` command. The normal `build/audit036` package was
checked and does not register it. `tests/failure_injection.test` passed 6/6
cases with each GGUF. Injected cases establish these binding-level outcomes:

| Injected point | Observed result |
|---|---|
| Before cloning a fork sampler | Tcl error; source info and active inference remain unchanged; destination remains unused and source can continue. |
| After the real KV-copy request, before metadata verification | Tcl error and global inference reset; registered sequences return to position zero, destination is not registered, and prefill can recover. |
| Sequence generation decode status | A synthetic nonzero status takes the normal reset path; both fork branches reset and can be prefilled again. |
| Joint `batch_step` decode status | A synthetic nonzero status resets all registered sequences; joint prefill and generation can resume. |
| Snapshot state restore into the temporary context | Tcl error; active context and position remain unchanged, and later normal restore is deterministic. |
| After temporary snapshot restore, immediately before context swap | Tcl error; the old active context remains usable and retains its position. |

These hooks do not simulate an allocator exhaustion, an exception inside
`llama_memory_seq_cp`, a partial native decode, or a backend that corrupts data
before returning failure. The decode hooks return a synthetic error before
calling `llama_decode`; only the post-copy hook performs the actual copy before
forcing the verification-failure branch. No real OOM test was attempted.
`valgrind` reported no definite leaks in the native sampler-history/clone test;
this does not prove absence of leaks in the full extension or backend.

### Memory methodology and observations

The temporary Tcl probe ran in fresh processes, three repetitions per model
and profile. Each process loaded one model and initialized a CPU context with
`n_ctx=512`, `n_seq_max=4`, `n_batch=64`, and two execution/batch threads. The
plain profile used temperature zero; the grammar profile used
`root ::= [0-9]{200}` and temperature zero. Measurements came from
`/proc/self/status` (`VmRSS` and cumulative `VmHWM`) after context load, prefill,
sampling, pending and materialized fork, four active sequence states, snapshot
save/restore/discard, branch deletion, `clear_cache`, and handle free. Values
below are medians across the three runs, in KiB unless bytes are stated.

| Model/profile | RSS after model+context init | HWM increase after one active sequence | HWM increase from one active sequence to four | Snapshot context-state payload | HWM increase over context init after restore |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B, plain | 1,383,968 | 4,060 | 1,504 | 952,249 B | 45,720 |
| Llama 3.2 1B, GBNF | 1,383,996 | 22,664 | 1,496 | 952,249 B | 56,736 |
| Qwen3 1.7B, plain | 1,498,136 | 4,016 | 1,780 | 2,870,281 B | 131,008 |
| Qwen3 1.7B, GBNF | 1,498,104 | 26,580 | 4,660 | 2,870,281 B | 139,784 |

The initial grammar sampling stage increased peak RSS/HWM substantially over
the matched plain stage: about 18.2 MiB for Llama and 22.0 MiB for Qwen3. This
stage includes sampler construction, grammar state, inference and allocator
effects; it is not an isolated measurement of the grammar sampler. The
increment from one active sequence to four combines copied logits, KV pages
touched/materialized, per-sequence sampler state and allocator behavior. It
must not be attributed solely to the `4 * n_vocab` logits vectors. The observed
fork-pending stage did not raise HWM at the 1 KiB reporting resolution; the
next decode/materialization raised it by roughly 0.5 MiB for Llama and 0.58 MiB
for Qwen3, consistent in scale with one float-logit vector but not proof that
the increase is exclusively logits.

Snapshot save reported native context-state payloads of 952,249 bytes for
Llama and 2,870,281 bytes for Qwen3. Snapshot restore temporarily constructs a
second context: cumulative peak HWM rose by about 38 MiB above the pre-restore
peak in plain Llama and 119 MiB in plain Qwen3, with lower but still substantial
increments in the grammar profiles. This is a process peak for the complete
restore operation, not a separately allocated-context measurement. After
snapshot discard, branch deletion and `clear_cache`, RSS generally did not fall
because freed allocations remained resident/reusable in the process allocator.
After handle free, process RSS fell to roughly 70–92 MiB; cumulative HWM, by
definition, remained high.

The context-init RSS includes mapped model data, Tcl/runtime libraries and
llama.cpp context/backend allocations; it is not a model-only footprint.
`VmRSS` varied during runs and cannot isolate native KV, sampler, logits or
reserved versus resident backend memory. HWM was more stable across repetitions
but is cumulative. The probe used four sequences, a short prompt and a 512-token
context; results are not a benchmark and do not extrapolate to large contexts,
larger models, other hardware, CUDA, AVX2 or AVX-512. The workload remained
well below the available host memory; no system OOM or memory-limit test was
performed.

## Handle lifecycle, mappings and sanitizers — Audit 037

### Ownership and release path

`llama::init` creates a model and context owned by a Tcl handle. The command
delete callback marks it closing and schedules `LlamaStateFree` through
`Tcl_EventuallyFree`, so an active operation/callback can finish before native
resources are destroyed. The state deleter releases its snapshot, per-sequence
sampler/logit records and tables, option storage, the handle sampler, then
`llama_free(ctx)` followed by `llama_model_free(model)`. This matches
context-before-model ownership. Temporary batches are released by their
command paths/guards; Tcl owns callback values and command objects.
`llama::free`, direct command deletion and interpreter teardown all use this
delete path. Existing lifecycle tests and the new repeated-cycle probe exercise
these paths, including callback deletion and multiple interpreters.

The package invokes `llama_backend_init` during extension initialization.
Backend registration and its loader state are process-level llama.cpp/ggml
state, not a resource that `llama::free` releases per handle. The selected
llama.cpp header describes `llama_backend_free` as an end-of-program operation
(currently used for MPI); tclllama does not call it on handle or interpreter
destruction. This audit found no basis for adding per-handle backend teardown.

### Repeated cycles and per-cycle RSS

The committed Linux probe is `tests/lifecycle_resources.tcl`. It records
`VmRSS`, cumulative `VmHWM`, PSS, anonymous/private/shared memory and GGUF
mapping count/size/RSS/PSS at every stage and cycle. It supports `plain`,
`infer`, `snapshot`, `multisequence`, `handles` and `interpreter` scenarios,
asserting that model mappings disappear after the last owner is deleted.
Model loads ran sequentially, never concurrently. Plain and inference modes
ran five initial cycles; inference and two-handle runs were extended to ten and
five cycles respectively. Snapshot, multisequence and interpreter scenarios
ran three cycles per model.

Values below are `VmRSS` after each cycle's final owner was released, in KiB
and cycle order. The probe also records HWM at each stage; HWM is cumulative and
does not decrease after release.

| Model | Scenario | Post-release RSS for each cycle (KiB) | GGUF mappings after release |
|---|---|---|---|
| Llama 3.2 1B | plain, 5 cycles | 67,752; 79,256; 79,256; 79,264; 79,264 | 0 each cycle |
| Llama 3.2 1B | infer, 10 cycles | 69,928; 79,632; 79,648; 83,424; 96,816; 79,644; 96,816; 79,644; 79,648; 79,644 | 0 each cycle |
| Llama 3.2 1B | snapshot, 3 cycles | 70,144; 96,896; 96,916 | 0 each cycle |
| Llama 3.2 1B | multisequence fork/generate, 3 cycles | 71,612; 79,772; 79,784 | 0 each cycle |
| Llama 3.2 1B | two handles, 5 cycles | 131,608; 189,536; 207,008; 229,320; 229,440 | 0 after both handles |
| Llama 3.2 1B | interpreter teardown, 3 cycles | 70,968; 96,996; 97,012 | 0 each cycle |
| Qwen3 1.7B | plain, 5 cycles | 55,824; 65,896; 65,900; 65,908; 65,908 | 0 each cycle |
| Qwen3 1.7B | infer, 10 cycles | 58,252; 67,380; 67,388; 67,408; 67,472; 67,476; 67,476; 71,772; 71,772; 71,776 | 0 each cycle |
| Qwen3 1.7B | snapshot, 3 cycles | 58,660; 67,340; 67,756 | 0 each cycle |
| Qwen3 1.7B | multisequence fork/generate, 3 cycles | 59,684; 69,280; 69,392 | 0 each cycle |
| Qwen3 1.7B | two handles, 5 cycles | 100,516; 134,152; 134,156; 134,156; 134,156 | 0 after both handles |
| Qwen3 1.7B | interpreter teardown, 3 cycles | 59,176; 67,532; 67,532 | 0 each cycle |

For one-handle `plain` workloads, RSS reached a small plateau after the first
cycle. Inference and interpreter paths fluctuated before settling into a narrow
range rather than rising monotonically through ten cycles. Two-handle runs
retained more anonymous RSS after release: Llama rose to about 229 MiB and
changed by only 120 KiB between cycles 4 and 5; Qwen reached about 134 MiB by
cycle 2 and stayed level through cycle 5. All final-owner releases had zero
GGUF mappings. These plateaus are compatible with allocator/runtime retention
or reuse, but RSS alone cannot prove absence of a leak or attribute retained
anonymous pages to a particular library.

### GGUF mappings and multiple handles

After one handle initialized, `/proc/self/smaps` showed one GGUF mapping, with
file-backed RSS/PSS of **1,282,476 KiB** for Llama and **1,321,616 KiB** for
Qwen. With two handles loading the same file, there were two mappings and
file-backed RSS doubled to **2,564,952 KiB** / **2,643,232 KiB** respectively.
Their summed PSS remained **1,282,476 KiB** / **1,321,616 KiB**, consistent
with the operating system sharing resident file pages across mappings. This
does not indicate that llama.cpp shared one native model object: each Tcl
handle loaded and owned its own model/context.

After freeing the first handle, one mapping remained and the second handle
successfully generated a token. After freeing the second, mapping count,
mapped GGUF RSS and mapped GGUF PSS all reached zero. The kernel may retain
clean file pages in its page cache after unmapping; that is separate from
process RSS and was not manipulated with `drop_caches`. `VmHWM` remains at the
largest value observed during a process even after current RSS falls.

### AddressSanitizer and LeakSanitizer

An isolated GCC 15.3.1 build used `-fsanitize=address`, debug symbols and frame
pointers, with `LD_PRELOAD=/usr/lib64/libasan.so.8` because the Tcl executable
is not instrumented. `ASAN_OPTIONS` enabled leak detection, halt-on-error,
allocator mismatch checking and strict string checks. The binding,
`schema_adapter.cpp` and llama.cpp `common/*.cpp` objects built by the Makefile
were instrumented. The prebuilt static `libllama.a`/ggml archives and Tcl
runtime were not. No AddressSanitizer or LeakSanitizer diagnostics appeared:

- `lifecycle.test`: 10/10 per model.
- `fork.test`: 10/10 per model.
- `snapshot_multisequence.test`: 7/7 per model.
- `failure_injection.test` in a separate ASan build with `TCLLLAMA_TESTING`:
  6/6 per model.
- Lifecycle probe: inference for 3 cycles, snapshot for 1, multisequence fork
  and generation for 1, and two handles for 2 cycles per model.

These checks cover the instrumented binding paths and their boundary calls;
they do not prove memory safety inside the uninstrumented static dependencies.
They did not include a real OOM or backend crash.

### Valgrind Memcheck

Valgrind 3.27.1 directly supervised `tclsh` for one plain model/context
init/free cycle per GGUF. Both runs completed with **0 definitely lost bytes,
0 indirectly lost bytes and 0 Memcheck errors**. Each reported **1,759,232
bytes in 102 blocks as possibly lost** and no suppressions. The inspected
possible-loss stacks point into Tcl interpreter allocation, script reading,
regexp and bytecode paths. A trivial Tcl-only process reported 453,632 bytes
possibly lost; loading the package without a model reported 1,447,936 bytes.
Those comparisons do not prove all possible-loss entries harmless, but no
definite loss was attributed to the binding. The model-cycle runs also showed
zero GGUF mappings after free. Under Valgrind, post-release RSS was around
280–300 MiB because Memcheck changes process memory; do not compare it with a
normal build.

The Valgrind model-cycle runs did not exercise inference or snapshots; those
paths were exercised under ASan/LSan and normal regressions. No leak
suppression was added.

### Audit 037 conclusion and limits

The measured model mappings disappear after the last owning handle/interpreter
is destroyed, and a second handle remains usable after the first is freed. The
tested one-handle workloads did not show sustained linear RSS growth through
ten inference cycles. Two-handle release leaves a larger anonymous-RSS plateau,
without a definite-loss report or remaining GGUF mapping. This is evidence
against a model-mapping leak in these runs, not proof that every native/backend
allocation is leak-free.

No production ownership or inference code required a correction. CUDA, AVX2,
AVX-512, other operating systems, other Tcl runtimes and llama.cpp/ggml built
with sanitizers remain untested. GPU/backend allocations are not represented
by these CPU RSS/PSS measurements.

## Independent sequence text output — Audit 038

`llama::sequence ... generate_text` was added as a synchronous text-returning
counterpart to raw-token `generate`. Both operations use the same per-sequence
sampler, saved logits and position tracking; the new path converts token pieces
incrementally with the existing `TextBuffer`, including UTF-8 boundaries, stop
IDs, callbacks and the established GBNF/JSON Schema marker-filtering rule.
Callback errors use the existing whole-context reset recovery. `batch_step`
remains token-oriented.

The dedicated `tests/sequence_text.test` passed **6/6** with Llama 3.2 1B
Instruct Q8_0 and **6/6** with Qwen3 1.7B Q4_K_M. It covers raw-token
compatibility, equality between raw-token detokenization and text output,
multibyte UTF-8 output reconstructed from multiple generated tokens and
matching raw-token detokenization, literal `<|im_end|>` grammar text, per-call
stop-ID behavior, invalid-grammar preservation and callback-error recovery. The marker
regression initially exposed that the text buffer was consulting the prior
options value after sampler configuration was committed; it now uses the
sequence's committed options. These are CPU observations on Gepeto and do not
establish behavior for every tokenizer or backend.

## Shared native model with independent contexts — Audit 039

The pinned llama.cpp API exposes `llama_model_load_from_file`,
`llama_init_from_model(model, params)`, `llama_free(context)` and
`llama_model_free(model)`. The binding now represents that lifetime directly:
an opaque model handle owns one native model reference, and every context
handle retains the model until after its own native context is freed. The
legacy `llama::init` creates a private model owner and one context, preserving
its former one-call contract.

The test-only `::tclllama::test::same_model` probe compares both the internal
owner and raw `llama_model*` addresses of two contexts. Its successful result
is direct evidence that the tested contexts use the same native model object,
not just the same GGUF path. It is not registered in production builds.

Context-local KV memory, samplers, logits, sequence tables, persistent
sampling options and snapshots remain in `LlamaState`. Context operations that
use the same model are serialized by an owner-level reentrancy guard. A
callback in one context cannot reenter a sibling context; this is not a
thread-safety or concurrent inference claim. Model and context Tcl handles
remain tied to the creating interpreter.

The shared-model suite exercises context creation and destruction, deterministic
interleaving against independent references, creating a replacement context
without reloading the model, model-free rejection while contexts are live,
model-command deletion with retained contexts, snapshot/sequence isolation,
callback reentrancy, and legacy initialization. Separate reports below record
the exact test totals for the normal and native-identity test builds. CPU-only
measurements do not generalize to accelerator backends. Each context still
allocates its own compute/KV resources; sharing the model object does not imply
that all memory is shared or that RSS falls by a particular amount.

The final normal build ran 24 Tcl suites with each local model: 124 cases,
117 passed, 7 skipped and 0 failed. The skips are the six test-only fault
injection checks and the test-only native pointer identity check. The dedicated
shared-model suite passed 12/12 with each model in a `TCLLLAMA_TESTING` build;
that build's pointer probe confirmed both contexts hold the same native
`llama_model*`. The same 12 tests also passed under the isolated ASan/LSan
build for each model with no sanitizer diagnostics. The binding and the
Makefile-built common/schema objects were instrumented; prebuilt llama.cpp/ggml
archives and Tcl were not, so these runs do not establish their internal memory
safety.

### Bounded RSS/PSS comparison

A one-run process-level comparison used the Llama 3.2 1B Instruct Q8_0 GGUF,
Linux `/proc/self/status`, `/proc/self/smaps_rollup` and matching entries in
`/proc/self/maps`. Both paths used two contexts with `n_ctx=512`; the shared
path loaded one model then created two contexts, while the legacy path created
two `llama::init` handles. No inference was run. Values are KiB; PSS is
proportional set size, not a count of model bytes owned by llama.cpp.

| Stage | Shared model: RSS / PSS / GGUF map entries | Two legacy handles: RSS / PSS / GGUF map entries |
|---|---:|---:|
| Fresh Tcl process | 11,416 / 6,581 / 0 | 11,380 / 6,567 / 0 |
| Model loaded (shared) or first handle | 1,348,236 / 1,342,918 / 1 | 1,366,560 / 1,361,268 / 1 |
| Second context/handle created | 1,385,224 / 1,379,942 / 1 | 2,728,676 / 1,440,900 / 2 |
| First context/handle freed | 1,368,884 / 1,363,562 / 1 | 1,429,804 / 1,424,504 / 1 |
| All contexts/handles freed, model owner still held | 1,352,488 / 1,347,166 / 1 | — |
| Model freed / all legacy handles freed | 70,012 / 64,690 / 0 | 130,932 / 125,632 / 0 |

The shared-model process retained one GGUF mapping while either context or the
explicit model handle remained alive; freeing the final model handle removed
that mapping. Two legacy handles created two mappings. Their RSS rose by about
1.30 GiB when the second handle loaded the same file, while PSS rose by about
78 MiB, consistent with the kernel sharing many clean file-backed pages across
the mappings. In this sample, two contexts with one model had about 61 MiB less
PSS than two model/context handles after both were created. This difference is
an observation for one model, one context size, one process per scenario and
one run; it does not isolate every allocation or predict other backends.
Context creation added roughly 18 MiB of RSS per context in the shared case,
consistent with separate context-side resources, but this was not a KV
population or inference measurement. After final release, both scenarios had
zero matching GGUF mappings; remaining process RSS is not proof of a leak and
may include Tcl/runtime or allocator-retained pages. No physical-memory or
performance claim is extrapolated from this single comparison.

## Native llama.cpp state APIs — Audit 040

The binding now exposes the pinned engine's full-context and per-sequence
state-buffer and file APIs. The commands use the same `llama_context` whether
it came from legacy `llama::init` or explicit shared-model `context_create`.
Byte-array round trips retain arbitrary byte values. Sequence restoration uses
llama.cpp's destination sequence remapping. Host `none` and `partial_only`
(`swa_only` alias) flags are exposed; `ON_DEVICE` is intentionally unavailable
because its buffer is a device resource with a different lifetime contract.
The tests exercise the flag interface on the two standard decoder fixtures;
they do not validate partial-state behavior on an SWA, recurrent or hybrid
memory model.

These native APIs do not serialize the binding's sampler chains, grammar
progress, sequence registry, Tcl options, per-sequence logits or terminal
flags. A successful native restore therefore marks the handle
`native_state_only`, and binding inference rejects the mismatched wrapper state
until `clear_cache` or restoration of a complete tclllama snapshot. Restore is
performed against a temporary context (and sequence restores start from a
native context copy), so tested validation failures preserve the active native
context. This uses extra memory during restore and does not establish
atomicity inside llama.cpp or atomic file writes. Upstream validates limited
metadata; portability across models, context configurations, backends or
llama.cpp revisions is not claimed.

There is a source/header scope discrepancy in this pin: the public header's
`llama_state_get_size` comment says context state covers logits, embeddings and
memory, while `llama_context::state_write_data` in the matching source writes
the model architecture string and the memory module state. That function does
not write the context output-logit or embedding buffers. The round-trip tests
verify native memory/position restoration, not output-buffer restoration;
logits and embeddings are therefore not claimed as preserved by this binding.

`tests/native_state.test` passed **12/12** for each of Llama 3.2 1B Instruct
Q8_0 and Qwen3 1.7B Q4_K_M. It covers memory and file round trips, sequence ID
remapping, binary data, invalid state preservation, callback busy guards,
shared-model context isolation and legacy API use.

The independent review of Tcltest skips reconciles the previous reports:
Audit 037's eight skips comprised six tests requiring the separate
`TCLLLAMA_TESTING` fault-injection build plus two chat checks in `compat.test`
and `strategies.test`. Those chat checks have an explicit constraint requiring
`TCLLLAMA_TEST_CHAT=1`; they were omitted in the 037 run. In the 039 run the
variable was enabled, and `compat.test` (8/8) and `strategies.test` (4/4)
passed. The seven skips reported in 039 were instead six fault-injection checks
and the native model-pointer identity probe, which is available only in the
test build. No test counts were adjusted to reconcile the reports.

The requested ASan/LSan run used an isolated instrumented extension. For each
GGUF, `sequence_text` passed 6/6, `streaming` 2/2 and `grammar` 4/4; the
additional `native_state` suite passed 12/12. No AddressSanitizer or
LeakSanitizer diagnostics were emitted. The Tcl executable and prebuilt
llama.cpp/ggml archives are not instrumented, so diagnostics do not cover their
internal memory accesses. This targeted run is not a general leak-freedom
claim.

The final normal build `build/audit040` passed all **25 Tcl suites** for each
fixture: **136 cases, 129 passed, 7 skipped, 0 failed** per model. In both
runs, six skips were the fault-injection cases requiring `TCLLLAMA_TESTING`,
and one was the native model-pointer identity check available only in that
test build. Chat constraints were enabled; no chat test was skipped. The four
native synthetic checks (sampler history, logit bias, sampling strategies and
schema adapter) also passed.

## Native state continuation — Audit 041

The 040 `native_state_only` guard was narrowed for states whose native memory
can be reconciled with the wrapper. Full-context restore is eligible only for
single-sequence (`n_seq_max == 1`) `llama` or `qwen3` decoder contexts without
recurrent, hybrid or SWA memory, with sequence 0 either empty or contiguous
from position zero and within `n_ctx_seq`. The binding derives the next
`n_past` from the native memory range and prepares a fresh sampler from the
handle's currently persistent options before swapping the temporary context.
The caller must decode new prompt tokens after restore to produce logits.
Session-file restore is eligible only when its returned token count equals the
restored position count.

The pinned public memory API exposes sequence position minima and maxima, not
an enumeration/count of every occupied cell. The binding checks a zero-based
range and capacity but cannot independently detect an interior hole in an
arbitrary externally produced native state. The continuation tests use states
created by tclllama's contiguous prefill/generation paths; this is not a
validation guarantee for every third-party native state producer.

For sequence restore with `-flags none`, an otherwise synchronized context can
reconcile only the destination sequence. Its position is read from the new
context; an independent sequence receives a fresh sampler using its previously
configured options (or the handle defaults), with logits and started/finished
flags cleared. `sequence prefill` with new token IDs must run before generation.
The same path works when the handle was created by legacy `llama::init` or by
`model_load`/`context_create` for single-sequence contexts. Multi-sequence
context-wide restore, partial-only restores, shifted/noncontiguous memory,
unsupported architectures and restores layered on an unreconciled native state
remain `native_state_only`.

`tests/native_state.test` passed **16/16** cases with each local CPU model:
Llama 3.2 1B Instruct Q8_0 and Qwen3 1.7B Q4_K_M. The cases verify context
byte-array restore followed by new-prompt generation through both legacy and
shared-model context handles, native context-file restore followed by
generation, and per-sequence state-buffer/file restore followed by additional
prefill. In independent mode the restored sequence starts with a fresh sampler
and becomes generatable after that prefill. The existing invalid-state test
confirms rejected short buffers leave the active context usable.

The same 16-case suite passed under the isolated ASan/LSan build with both
GGUFs, with no sanitizer diagnostics. The Tcl extension and local converter
objects were instrumented; the llama.cpp/ggml static archives and Tcl runtime
were prebuilt and therefore are outside sanitizer coverage. This is directed
ownership coverage for native-state restore, not a blanket memory-safety or
leak-freedom claim.

The final normal build passed all **25 Tcl suites** with both local GGUFs:
**140 total, 133 passed, 7 skipped, 0 failed** per model. Six skips are the
fault-injection cases that require the separate `TCLLLAMA_TESTING` build; the
seventh is the shared-model native-pointer identity probe, also test-build
only. The four native synthetic checks (sampler history, logit bias, sampling
strategies and schema adapter) passed. These runs use the local Linux CPU
backend and do not establish GPU or AVX2/AVX-512 coverage.

These results demonstrate continued inference from native KV plus newly
decoded input; they do not prove restored logits, sampler history, grammar
progress, penalty/DRY history, RNG state, or identical post-restore output.
Native file tokens are checked against restored position length before
enabling this path. Sequence state buffers do not carry a token list, so the
binding validates the native position range but requires the caller to provide
new prefill tokens. The temporary-context strategy preserves the active
context through validation and sampler preparation failures; atomicity of
llama.cpp operations themselves is not claimed. Tests were run on Gepeto's
Linux CPU build only; GPU, AVX2/AVX-512, recurrent/hybrid/SWA memory and other
architectures remain unverified.

## Native adapters and control vectors — Audit 042

The LoRA ownership fixture is an upstream GGUF with adapter metadata and **zero
tensors**. It exercises actual native adapter objects, attachment to two
contexts of one model, explicit-release rejection, renamed/deleted commands,
deferred model release, context/snapshot replacement and interpreter teardown.
It is not a trained adapter and does not validate learned LoRA inference.
No compatible trained LoRA fixture was found in the local model data; none was
downloaded. aLoRA invocation tokens are queryable, without activation scheduling.

The separate native `control-vector-test` uses identical token input in two
contexts. Setting a vector with component 0 equal to 1 on layer 1 changed the
vocabulary-logit L1 sum by approximately 1352.15 (Llama 3.2 1B Q8_0) and 6352.29
(Qwen3 1.7B Q4_K_M). Removing the vector and clearing/redecoding the controlled
context yielded L1 difference 0 from the reference on both CPU models. This is
numerical evidence of native application/removal, not a semantic quality test
or performance benchmark. Tcl tests additionally cover dimension checks,
replacement contexts and configuration isolation with zero vectors.

Directed ASan/LSan runs of `model_controls.test` passed 7/7 cases per model with
no diagnostics. The extension/converter objects were instrumented; prebuilt
llama.cpp/ggml archives and Tcl were not. Native allocation failures were not
injected. Preparing adapter/vector changes on a temporary context preserves
the active context on preparation failure; arbitrary backend failures and
other model architectures are not universally covered. GPU execution was not
tested. Additional model flags were exercised on the CPU-compatible setting
only; enum validation does not establish support for other backends or MTP.

The final normal build completed all 26 Tcl suites per model: 147 cases,
140 passed, 7 skipped, 0 failed. Six skips require TCLLLAMA_TESTING fault
injection; one is the test-build-only native model identity probe. The updated
shared-model message assertion was rerun successfully on Llama and included
in Qwen's final run. No functional source changes followed the builds used for
these regressions and the directed sanitizer runs.

## Device arrays and typed load overrides — Audit 043

The pinned llama_model_params API accepts explicit NULL-terminated devices,
float tensor_split arrays and empty-key-terminated typed metadata overrides.
The wrapper retains all three arrays with the model owner through the final
shared context. The engine copies tensor_split internally in this revision;
retention also avoids dangling parameter pointers for fields the engine stores.

The new model_loading suite has four cases per GGUF. It tests selection of the
registered CPU, empty offload selection, finite and zero split weights, typed
overrides, exact int64 endpoints, invalid/duplicate IDs and keys, string/key
bounds, NUL rejection, invalid numeric ranges/types and repeated load/free.
An architecture-specific context_length override set effective n_ctx_train to
2048 on both Llama 3.2 1B and Qwen3 1.7B; two contexts from one loaded model
remained independent. This verifies one effective metadata property; other
unused overrides need not appear in metadata strings and can be ignored by
the upstream loader. Override acceptance is not proof of arbitrary metadata
changes or compatibility with every GGUF key/type.

Directed ASan/LSan runs passed 4/4 cases on each local model without diagnostics.
The binding/converter objects were instrumented, while prebuilt llama.cpp/ggml
static archives and the Tcl runtime were not. No native allocation failures
were injected. Failed loads and invalid options return errors without exposing
a partially initialized model handle.

Selection validation and CPU inference do not validate GPU placement,
CUDA/Vulkan/ROCm, tensor parallelism or nondefault mlock/direct-I/O behavior.
No Tcl callback is invoked by the loader: progress callbacks and backend buffer
pattern overrides were deliberately left unexposed pending a safe contract.
Omitted n_gpu_layers now preserves native -1; explicit zero remains available
for CPU-only weight loading. The change was observed only on CPU hardware.

Final normal regressions passed all 27 Tcl suites with each GGUF: 151 cases,
144 passed, 7 skipped, 0 failed per model. Six skips require the special
fault-injection build and one requires its native model-identity probe. The
final suite includes the missing-file failure path with prepared native arrays;
the affected suite and directed sanitizer runs were repeated after that case
was added. Production source remained unchanged through these final runs.

## Advanced sequence-memory edits — Audit 044

The pinned header claims that every position between seq_pos_min and seq_pos_max
is present. In the conventional cache, seq_rm iterates cells in [p0,p1) and can
remove interior positions. The new integration test prefills positions 0..5,
removes [2,4), observes unchanged extrema 0/5 and a smaller serialized sequence
state. Min/max therefore cannot validate continuity after arbitrary deletion.
The binding deliberately marks such edited memory native_state_only rather
than treating max+1 as a proven contiguous prefix length.

The tests also exercise suffix/full deletion, full cross-stream copy, partial
prefix copy in unified memory, keep across streams, negative shifts and whole
position division. All admitted continuations decode new input before sampling;
no text equality with prior sampling is claimed. Position transforms are limited
to one populated sequence to avoid shared-cell side effects, and require native
can_shift. Deferred copies/shifts block export and snapshot save until decode.
The legacy snapshot path needed an added pending-update guard, and successful
legacy prefill now clears the pending flag. Empty legacy prompts while pending
are rejected before a native failure could trigger the old reset policy.

Protected edits copy native context state and reapply LoRA/control-vector
settings. New model_controls checks exercise memory edits while a zero control
vector or tensor-free synthetic LoRA is attached; this does not add trained
LoRA validation. No global RSS campaign or new backend was introduced.

Final normal regression: all 28 Tcl suites completed with each local GGUF,
160 cases, 153 passed, 7 skipped, 0 failed per model. Six skips require the
special fault-injection build; one requires its native model-identity probe.
Directed ASan/LSan runs passed 9/9 memory_advanced plus 7/7 model_controls cases
per model without diagnostics. Prebuilt llama.cpp/ggml and Tcl were not
instrumented. The affected message assertions in fork/fork_integrity were
rerun, and the updated snapshot assertion was included in the complete run.
Production source did not change after compiling these final builds.

## Context options and model features — Audit 045

The local llama/qwen3 decoder models exercise explicit output caps, causal/no
pooling modes, disabled flash attention, linear RoPE, training-scale YaRN,
float32 KV storage and native performance collection. Tests create two contexts
from one model with different settings and verify that snapshot restore, native
state restore and protected suffix removal preserve the requested recipe.
Importing an f32 cache state into an f16 context is rejected by the native reader
on a temporary context, leaving the destination at its prior position.

The initial YaRN probe revealed and reproduced an upstream model-wide mutation:
llama_init_from_model changes model->hparams.n_ctx_train when a YaRN context uses
a scale unlike training scale (src/llama-context.cpp). The final binding rejects
this before allocation/retention and checks that an existing sibling's model
property remains unchanged. Training-scale YaRN executes; custom-scale YaRN is
not claimed as independent or supported. No upstream source was modified.

New model fields come from native getters, including default template omission
when unavailable. Both tested GGUFs have a template; missing-template models,
MTP, encoder-only, recurrent/hybrid/SWA/diffusion models and long-context quality
were not functionally characterized. GGUF arrays were investigated through
native gguf_init_from_file, type/array getters and gguf_free; a full reader needs
a separate ownership/type/size contract and was not added.

Final production builds passed all 29 Tcl suites for each local GGUF: 164 cases,
157 passed, 7 skipped, 0 failed per model. Chat coverage was enabled. The skips
are six fault-injection cases requiring the special test build and one native
model identity check requiring test instrumentation; they are not approvals.
All four existing native checks passed. Directed context_configuration ASan/LSan
runs passed 4/4 cases per model without diagnostics. Only the binding and local
converter objects were instrumented; prebuilt llama.cpp/ggml and Tcl were not.
No production code changed after compiling the final normal/instrumented builds.

## Cooperative control — Audit 046

The existing paths sampled/accepted before checking EOG, CONTROL and stop IDs,
and text callbacks ran before decode. These policies remain; an ordinary token
returned by the control path has been decoded before the next poll. A false
control result preserves the sampler and context. A callback exception follows
the prior reset policy; callbacks may already have observed a partial stream.
The new details dictionary describes only normal completion, not every error
before inference begins. No native abort is enabled.

A Unicode probe without control reproduced a pre-existing Tcl 8.6 supplementary
literal/grammar transport limitation: a literal U+1F600 yielded replacement
characters while GBNF `\U0001f600` produced UTF-8 f09f9880. The integration test
uses native ASCII GBNF escapes and asserts exact UTF-8 bytes for accented text,
the supplementary scalar, a literal stop marker and a CJK scalar across snapshot
restore. This does not claim general Tcl-string normalization was corrected.

The existing synthetic stream_buffer test also passed: split UTF-8 scalars,
incomplete-scalar finalization, split complete markers and incomplete marker
prefixes. It exercises the same finish path used on cancellation. Real-model
tests cover partial ASCII streaming cancellation and Unicode continuation after
immediate cancellation; they do not force a GGUF to stop inside a scalar.

Directed ASan/LSan cancellation tests passed 14/14 on each local GGUF with no
diagnostics (detect_leaks, halt_on_error, alloc_dealloc_mismatch and
strict_string_checks enabled). Fresh binding objects were instrumented;
unchanged local converter objects reused the previous instrumented build.
The static llama.cpp/ggml archives and Tcl runtime remain uninstrumented.
This does not assert complete leak freedom or backend memory coverage.

Final production regression: 30 Tcl suites and 178 cases per local model,
171 passed, 7 skipped and 0 failed on each. Chat was enabled. Six skips require
the special fault-injection build; one native-identity check requires test
instrumentation. All four native checks and the additional stream-buffer
synthetic test passed. No production source or test changed after these final
normal/instrumented runs. CPU Gepeto results do not validate other backends.

## Vocabulary queries — Audit 047

The new Tcl command forwards public llama.cpp vocabulary getters and copies
strings/IDs into Tcl objects during the call. No global vocabulary copy or
context mutation is introduced. The two existing local GGUFs exercise the
available sentinel/suppression behavior; missing-special and null-list behavior
remain governed by native return values and are not inferred for untested model
families. Vocabulary `text` is not a promise of valid display UTF-8.

## Tokenization options and native detokenization — Audit 048

The pinned header defines negative return values from `llama_tokenize` and
`llama_detokenize` as the required capacity when the supplied output buffer is
short, and reserves `INT32_MIN` for tokenization overflow. The public
implementation constructs tokenization input with the explicit Tcl byte
length, so embedded NUL is included instead of terminating the input early.
The old token-piece detokenizer remains available with its prior call form;
the new native path validates token IDs before calling upstream and returns a
ByteArray so arbitrary native bytes are not silently converted as UTF-8.
GGUF-backed behavior is recorded by `tests/tokenization.test`; no claim is made
about vocabularies or byte sequences outside the exercised local models.
