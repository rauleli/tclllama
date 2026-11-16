# Implementation roadmap

This roadmap reflects the internal development priorities while keeping the
public September review TODO (`TODO.md`) distinct. The internal TODO and its
historical backup remain outside this repository and are not replaced here.

## Binding: primary line

### Implemented and exercised

- Handle-based model lifecycle, validation and deletion cleanup.
- Tcl validation for initialization, generation and chat arguments.
- CPU model loading, tokenization/detokenization, generation and chat.
- Context continuation/reset, callbacks, stop IDs, UTF-8 buffering and telemetry.
- Model/context parameters and GGUF metadata through `llama::info`.
- Per-handle temporary snapshots.
- Sequence 0 memory position inspection and guarded suffix truncation, with
  sampler reset and snapshot preservation.
- Native sampler options including Mirostat, logit bias, Typical, DRY and XTC.
- GBNF and constrained JSON generation through the native JSON Schema converter.

The evidence is bounded by the models and configurations in
[CHARACTERIZATION.md](CHARACTERIZATION.md). This is not a claim that every
llama.cpp API, model family or backend has been exposed.

### Remaining core work

- Continue a systematic, evidence-based inventory of llama.cpp capabilities
  that are useful and viable through Tcl.
- Investigate additional context, sequence/KV-cache, batching and multi-sequence
  APIs, speculative decoding, multimodality, metrics and controls only when the
  API and use case are characterized.
- Revisit stop/cancel semantics, option persistence, sampler history and
  context overflow where a concrete contract decision is needed.
- Complete packaging, dependency/link strategy, TEA integration and version
  metadata before a pre-release.
- Expand model/template and backend coverage. GPU behavior remains unverified.

### Public September TODO: evidence-backed status

The public `TODO.md` is retained as a review checklist; its priorities have not
been rewritten. The following items now have implementation or evidence in this
checkout:

| Public TODO area | Status at this baseline |
| --- | --- |
| Handle-based API and examples | Reconciled in `API.md` and `README.md`; current commands use `llama::` and handles. |
| Package/code version distinction | Explained in API, README and baseline docs; final runtime/package release metadata remains open. |
| Documented sampling/initialization options | Current supported options are documented; obsolete singleton examples/tables are removed. |
| Handle identity, cleanup and foreign/stale handle rejection | Implemented and exercised in lifecycle tests. |
| Argument validation and atomic configuration | Implemented and exercised in argument tests. |
| State, reset, sampler and stop behavior | Characterized for the tested models; prompt penalty history and broader semantics remain open. |
| llama.cpp pin and representative model checks | Recorded in `BASELINE.md` and `CHARACTERIZATION.md`. |
| Streaming, UTF-8, GBNF and structured-output tests | Implemented and exercised for the two fixtures; coverage is not universal. |
| Decision/characterization records | Added as `DECISIONS.md` and `CHARACTERIZATION.md`. |
| Sanitizer/Valgrind and context-overflow characterization | Still pending. |
| Multi-thread safety and concurrent handle use | Still pending; interpreter ownership is not a thread-safety guarantee. |

This status does not mark or reprioritize the original checklist.

## Auxiliary Tcl utilities

The Ollama registry manager already has source procedures for downloading,
listing and deleting local models, showing model information and resolving
paths. Add dedicated tests and user-facing examples before treating the utility
contract as fully characterized. This utility remains separate from native
model loading through `llama::init`.

## Later lines and optional tools

After the primary binding work:

1. Add embeddings through llama.cpp and develop RAG workflows, drawing on
   tclembedding experience without requiring compatibility between libraries.
2. As a separate line after embeddings, investigate llama.cpp STT.
3. Add optional user interfaces: terminal chat, Tcl/Tk desktop chat and Apache
   + Rivet WebUI. These should use the Tcl binding rather than add another
   inference engine.
4. As the last compatibility stage, design an optional resident service and
   implement the useful subset of Ollama's API for external clients. The
   binding must remain usable without this service.

The native binding, Tcl utilities, user interfaces and optional network service
are separate project components; none should be documented as implemented
before it exists and is verified.
