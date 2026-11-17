# Changelog

This file records verified changes to the implementation. It does not imply
that the consolidated public release has been published.

## [Unreleased] — target public version 1.0.0

### Added

- Native engine introspection through `llama::engine_info`, built-in chat
  template enumeration through `llama::chat_templates`, and default/named
  model template lookup through `llama::model_template`; existing package
  version reporting and `llama::chat` behavior are unchanged.
- Additional context output/RoPE/YaRN/cache options, requested/effective context
  dictionaries and native model characteristics; unsafe custom YaRN scales
  are rejected to protect shared-model metadata.
- Native sequence-memory rm/cp/keep and restricted add/div operations, with
  protected context replacement, fresh affected sampling and pending-update
  serialization guards on traditional and multi-sequence handles.
- Explicit native device selection, tensor split arrays and typed GGUF load
  overrides with buffers retained by the shared model owner. Omitted
  n_gpu_layers now preserves the native -1 default; use explicit 0 for CPU.
- Native LoRA handles, context-local adapter sets and numeric control vectors,
  with model references and configuration reapplied to replacement contexts.
- Scalar GGUF metadata enumeration, runtime thread controls/synchronization and
  additional native model/context creation flags.
- Handle-based model and context operations, argument validation and
  deterministic cleanup paths.
- Context continuation and reset behavior, callback streaming, UTF-8 buffering,
  stop-ID handling and telemetry.
- Initialization options for context batch/thread settings and selected model
  loading parameters.
- Extended model metadata through `llama::info`.
- One temporary in-memory inference snapshot per handle, including context,
  sampler and persistent sampling configuration.
- GBNF-constrained generation and a restricted JSON Schema conversion path using
  llama.cpp's native converter.
- Persistent token-ID logit bias, including explicit token suppression.
- Native Typical, DRY and XTC sampling options with validation of incompatible
  Mirostat combinations.
- Native dynamic temperature, Top-n-sigma and Adaptive-P sampling options.
  Defaults preserve the existing chain. Clone-dependent Adaptive-P operations
  are rejected because the pinned engine clone does not copy its advanced RNG.
- Synchronous multi-sequence prefill through one native `llama_decode` call,
  retaining the final logits independently for each sequence in sampling mode.
- Shared-round generation through `llama::batch_step`: independent sequence
  samplers select one token each, and ordinary tokens are submitted together
  in one native decode call.
- Whole-handle in-memory snapshots for supported multi-sequence decoder
  contexts, including sequence registrations, positions, cloned samplers,
  persistent sampling options, saved logits and terminal flags.
- Whole-prefix sequence forks after independent sampling has begun, cloning
  the sampler state, cached logits and persistent options for the new branch.
- Synchronous text generation for independent sequences through
  `llama::sequence ... generate_text`, with callbacks, UTF-8 reconstruction,
  token stop IDs and the established constrained-text marker policy.
- Explicit `llama::model_load`/`llama::model_free` and
  `llama::context_create`/`llama::context_free` commands, allowing multiple
  independent native contexts to retain one loaded `llama_model`. The legacy
  `llama::init`/`llama::free` path remains available.
- Bindings for llama.cpp's native context and sequence state APIs, including
  binary Tcl byte arrays and native session/sequence files. Compatible
  single-sequence context restores and full per-sequence restores now
  synchronize positions and permit continuation after new input is decoded,
  using fresh samplers rather than claiming exact sampler-state restoration.
- Tcl and native regression suites exercised against Llama 3.2 1B Q8_0 and
  Qwen3 1.7B Q4_K_M on CPU.

### Changed

- The documented API now matches the namespaced `llama::` commands and explicit
  per-interpreter handles.
- Sampling options are documented as persistent per handle, matching the
  current behavior.
- Build, model and compatibility claims are limited to the verified
  llama.cpp baseline and tested CPU configurations.
- The current local build is documented separately from the unverified legacy
  TEA packaging path.

### Fixed

- Updated the binding to llama.cpp revision
  `1719dbb0ee7bdaf56de083079d3613bb670160a5` and resolved compile/runtime
  incompatibilities covered by the adaptation tests.
- Hardened validation and failure recovery for handles, arguments, sampling
  candidates, callbacks, snapshots and constrained generation.
- Corrected the two verified defects in `library/ollama_registry.tcl` related
  to file output and download timer reset.

### Documentation

- Added `DECISIONS.md`, `CHARACTERIZATION.md`, `BASELINE.md`, `HISTORY.md` and
  `ROADMAP.md` to separate contracts, evidence, baseline, technical evolution
  and future work.
- Clarified that internal code/package progression `7.5` is not the complete
  library version; the intended public release target is `1.0.0`.
- Reconciled the README and API guide, retaining corrected usage examples and
  marking unsupported platform/backend claims as unverified.

## Historical repository record

The inherited changelog contained a `1.0` entry dated 2024-12-21. That entry
predates the consolidated binding and its broad feature claims are not treated
as evidence for the current implementation. The runtime/package value `7.5`
tracks historical code progression; the target consolidated public version is
`1.0.0`. Runtime/package metadata and release packaging remain to be reconciled
before publication.

- Add call-local `-control` and `-details` to generation/chat and independent
  raw/text sequence generation, preserving legacy return values and callbacks.
  Controlled cancellation keeps decoded state; errors retain reset semantics.
- Add `llama::vocab` read-only queries for native token properties, special IDs,
  vocabulary type/size and model-suppressed tokens.
- Add explicit `-add_special` and `-parse_special` options to tokenization,
  retaining the previous defaults, and an opt-in `llama_detokenize` path that
  returns exact bytes as a Tcl ByteArray while preserving legacy detokenization.
