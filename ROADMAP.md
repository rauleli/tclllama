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
- Explicit native model handles shared by independent context handles, while
  retaining the existing combined `llama::init`/`llama::free` API.
- Per-handle temporary snapshots.
- Sequence 0 memory position inspection and guarded suffix truncation, with
  sampler reset and snapshot preservation.
- Native sampler options including Mirostat, logit bias, Typical, DRY and XTC.
- GBNF and constrained JSON generation through the native JSON Schema converter.
- Native llama.cpp context and sequence state transfer through byte arrays and
  native session files; safe single-sequence restores can synchronize the
  position and continue after new input is decoded, with fresh sampling state.

The evidence is bounded by the models and configurations in
[CHARACTERIZATION.md](CHARACTERIZATION.md). This is not a claim that every
llama.cpp API, model family or backend has been exposed.

### Remaining core work

- Continue a systematic, evidence-based inventory of llama.cpp capabilities
  that are useful and viable through Tcl.
- Continue investigating additional context, sequence/KV-cache, batching and
  multi-sequence APIs, speculative decoding, multimodality, metrics and controls
  only when the native API and use case are characterized.
- Consider whether future state transfer should include binding-owned sampler
  and sequence metadata. Current native state commands follow llama.cpp's
  contract and are not portable conversation snapshots; tclllama's complete
  snapshots remain temporary in-memory state for one live handle.
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
| One native model reused by independent contexts | Implemented with explicit model/context handles; native pointer identity and serial context isolation are tested in the test build. |
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

Single-sequence token batch prefill is available through `llama::prefill`.
Multiple independent handles remain distinct contexts; shared-context sequence
support is described below. General prefix branching remains future work.

An initial shared-context multi-sequence prefill mode is available via
`n_seq_max` and `llama::sequence`. Opting into `sequence_sampling` adds independent
raw-token generation with sequence-owned samplers/options and saved logits.
`llama::batch_prefill` can now submit prefill tokens for distinct sequences in
one native decode call. `llama::batch_step` now samples one token per selected
independent sequence and submits ordinary tokens together in one native decode
call. `llama::sequence ... generate_text` returns text and supports UTF-8
callbacks while advancing one independently sampled sequence at a time.
In-memory whole-handle snapshots capture and restore supported multi-sequence
contexts, including registrations, positions and independent sampler/logit
state. Text output for `batch_step` rounds, mid-generation sampling
reconfiguration, a request scheduler, continuous batching, broader prefix
branching and snapshots for other memory families remain pending.

Whole-prefix fork for `llama` and `qwen3` now also supports active independent
samplers by cloning sampler state, options and logits. Forks of finished
sequences or arbitrary KV ranges, and forks for other architectures, remain unavailable;
physical sharing and memory savings are not established. Multisequence
snapshots remain temporary, in-memory state limited to the tested decoder
memory family rather than portable session files.

Native LoRA handle/application APIs and numeric control vectors are exposed.
Tensor-free synthetic adapters cover ownership; inference with trained LoRA
weights remains unverified pending a compatible fixture. Automatic aLoRA
invocation processing, upstream control-vector file loading, buffer overrides
remain separate work. Restricted native memory editing is covered below. Device/tensor arrays
and metadata load overrides are covered below. No scheduler or asynchronous execution is provided.

Explicit registered-device selection, tensor distribution weights and typed
GGUF load overrides are exposed. CPU selection/validation is characterized;
GPU distribution remains hardware validation work. Tcl load-progress callbacks
and tensor buffer-pattern overrides remain pending and are not silently mapped
to other mechanisms.

`sequence ... memory` now covers native range removal, prefix copying, exclusive
keep and restricted whole-sequence add/div transforms. General shared-cell
transforms, partial cross-stream copies and safe automatic reconciliation of
arbitrary holes remain pending. These operations do not provide a scheduler,
conversation manager or portable persistence format.

Context creation now exposes output limits, native attention/pooling defaults
within the causal generation contract, Flash Attention enum, RoPE/YaRN scalars,
f16/f32 KV types and no_perf. Requested values and public effective getters are
separate. Training-scale YaRN is allowed; custom YaRN scale remains blocked due
to a model-wide mutation in the pinned engine. Quantized/BF16 KV, recurrent
rollback/MTP/MoE tuning, embedding pooling and a typed GGUF array reader remain
pending. Model characteristic detection does not validate those inference paths.
