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
recovery policy. Prefix-sharing and branch separation are not implemented or
claimed. Native `n_ctx_seq` and allocation rounding determine effective capacity.
The functional sequence tests configure two sequences; the validation limit of
64 is not a claim that a 64-sequence workload has been tested.
On the final CPU build, all 15 Tcl suites passed with Llama 3.2 1B and Qwen3
1.7B: 59 tests per model, no failures or skipped tests, including four new
sequence tests. Native sampler-history, logit-bias, sampling-strategies and
schema-adapter synthetic checks also passed.
