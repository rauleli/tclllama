# Changelog

This file records verified changes to the implementation. It does not imply
that the consolidated public release has been published.

## [Unreleased] — target public version 1.0.0

### Added

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
- Synchronous multi-sequence prefill through one native `llama_decode` call,
  retaining the final logits independently for each sequence in sampling mode.
- Shared-round generation through `llama::batch_step`: independent sequence
  samplers select one token each, and ordinary tokens are submitted together
  in one native decode call.
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
