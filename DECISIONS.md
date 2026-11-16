# Decisions and open questions

This file records the current Tcl binding contracts separately from experiments.
The implementation source and [API.md](API.md) remain the operational authority.

## Current contracts

| Topic | Current contract | Basis |
| --- | --- | --- |
| Project and command identity | The project is named `tclllama`; its public Tcl commands are in the `llama::` namespace. Model operations use explicit opaque handles. | `generic/tclllama.c`, API and lifecycle tests. |
| Handle ownership | A handle belongs to the creating interpreter. Explicit free, Tcl command deletion/rename-to-empty and interpreter teardown release its native resources. Foreign, stale and cross-interpreter handles are rejected. | Handle delete callback and `tests/lifecycle.test`. |
| Argument errors | Unknown, duplicate, malformed and out-of-range options return Tcl errors. Sampling configuration is parsed before it replaces the active configuration. | Parser implementation and `tests/arguments.test`. |
| Initialization | The context size is positional, defaults to 4096 and is accepted from 512 to 32768. Model and context options are accepted through `-options`; GPU-related options are not evidence of GPU support. | `ParseInitOptions`, `Llama_Init_Cmd`, `tests/arguments.test`. |
| Sampling persistence | Sampling dictionary values and `num_predict` persist on the handle. `-max_tokens` cannot be combined with `-options num_predict`. A reset restarts inference/sampler state while keeping configured values. | Parser, `CommitCallSampling`, continuity and sampler tests. |
| Generation and chat state | `generate` continues the existing context unless `-reset` is true. `chat` formats the supplied message list and resets inference state for each call. A later `generate` continues from the chat context. | `Llama_Generate_Cmd`, `Llama_Chat`, `tests/continuity.test`. |
| Failure recovery | Input/configuration validation errors preserve active inference state. Decode or callback failures clear inference state when the handle remains live, allowing reuse. Partial callback output already delivered cannot be recalled. | Code and argument, lifecycle and streaming tests. |
| Snapshots | Each handle owns at most one temporary in-memory snapshot. It captures native context state, token position, cloned sampler and persistent sampling values. Restore builds a compatible temporary context and swaps only after a successful read. | Snapshot commands and `tests/snapshot.test`. |
| GBNF and JSON Schema | Grammar is persistent. JSON Schema uses llama.cpp's native converter through a deliberately restricted accepted subset. The modes are mutually exclusive. Early termination may return incomplete output. | API tests and source adapter. |
| Logit bias | Bias maps token IDs to finite adjustments or `suppress`. Grammar filtering precedes bias; bias cannot restore grammar-excluded tokens. Empty/invalid candidate sets return controlled errors in the guarded sampling path. | Native synthetic tests and model regressions. |
| Typical, DRY, XTC | The three native strategies default to disabled. Their combination with Mirostat 1/2 is rejected because the selected common Mirostat chain does not apply them. | `tests/strategies.test`, `tests/sampling_strategies.cpp`. |
| Version labels | Runtime/package value `7.5` is the historical code/package progression. It is not the complete library's public release number. The planned consolidated public release is `1.0.0`. | `Tcl_PkgProvide`, `llama::version`, project release plan. |

## Open decisions and limits

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
