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
