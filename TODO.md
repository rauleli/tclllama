# TODO — tclllama revalidation and hardening

This file records follow-up work identified during a repository review on **2026-09-30**.

The intent is not to rewrite the project indiscriminately. The current implementation already contains substantial functionality; the next pass should establish the real contract, verify lifecycle and state semantics, and then evolve the binding in small evidence-backed slices.

## Working method

Treat this file as a sequence of questions to resolve, not as permission to implement everything at once.

- Work in **small, independently reviewable slices**.
- **Characterize current behavior first**, then decide the intended contract, then modify code.
- Do not mix lifecycle, parser behavior, sampler semantics, upstream compatibility and documentation changes in one slice unless the dependency is unavoidable.
- A capability should not become public merely because llama.cpp exposes it. Identify the Tcl application or utility that would use it and the operational reason first.
- Separate **contract** from **characterization**: decisions belong in `DECISIONS.md`; measured/model-specific evidence belongs in `CHARACTERIZATION.md`.
- Preserve exact evidence for claims: upstream commit, model checksum, backend, host, parameters, exact output and tests.
- Do not silently broaden scope while fixing a nearby issue.

## Review follow-up — 2026-09-30

### API and documentation reconciliation

- [x] **Reconcile the public API with the implementation.** The current handle-based `llama::` contract and executable examples are documented in README, API, build and contribution guidance.
- [x] **Normalize development package versioning (Auditoría 064).** Tcl/TEA were verified to accept `0.1.0`; active package metadata, install paths, scripts, tests and current usage documentation use it. Historical `7.5` remains provenance, and public `1.0.0` is reserved for the joint tclllama/`tclllama-server` release.
- [x] **Audit documented options against actual parsing and upstream llama.cpp.** Current initialization and sampling options are documented against the pinned implementation; legacy unsupported options were removed from the current API reference.
- [ ] **Decide the future option surface deliberately.** The current implementation accepts an options object consumed through Tcl's keyed-value facilities. Before preserving or replacing that contract, identify consumers, compatibility requirements and migration cost. If the public API is revised, prefer explicit `-option value` parsing with deterministic handling of unknown, duplicate, missing and invalid values rather than carrying forward the current structure by inertia.
- [x] **Separate contractual behavior from observed/model-specific behavior.** Current contracts and model-specific evidence are separated in `DECISIONS.md` and `CHARACTERIZATION.md`.

### Handle identity, ownership and lifecycle

- [ ] **Stop relying on pointer-derived handle names as durable identity.** Current handles are generated with `"llama%p"`. Revisit identity using an interpreter-local monotonic scheme or another collision-resistant mechanism, informed by the handle-reuse issue already characterized in tclwhisper.
- [ ] **Give handle commands an explicit delete callback.** Make explicit `llama::free`, `rename $handle {}`, command deletion and interpreter destruction converge on one cleanup path.
- [ ] **Make cleanup idempotent and ownership explicit.** Define and test ownership/order for sampler, context, model and any future per-handle allocations.
- [ ] **Strengthen handle validation.** Do not accept an arbitrary Tcl command merely because `Tcl_GetCommandInfo` succeeds. Verify that the command was created by tclllama and that its client data and command identity are internally consistent.
- [ ] **Test stale, renamed, deleted and foreign-command handles.** Invalid handles must fail deterministically without dereferencing unrelated `ClientData`.

### State and option semantics

- [ ] **Decide which generation options are per-call and which belong to the handle.** `apply_options()` currently mutates persistent `LlamaState`; determine whether values such as temperature, sampling parameters, seed and token limits should persist into later calls or revert to defaults when omitted.
- [x] **Define `generate` versus `chat` state semantics.** Document and test context persistence, `-reset`, chat reset, system-message handling, and transitions between `generate` and `chat` on the same handle.
- [ ] **Prevent silent state leakage between calls.** Once the intended semantics are chosen, add regression tests proving omitted options and previous calls cannot change later behavior unintentionally.
- [ ] **Review sampler rebuild semantics.** Verify the order and compatibility of temperature, top-k, top-p, min-p, penalties, Mirostat and distribution sampling against the selected llama.cpp version.

### Argument parsing and Tcl behavior

- [x] **Harden option parsing.** Missing values, unknown/duplicate options and failed Tcl conversions return deterministic Tcl errors without partial sampling configuration changes.
- [ ] **Define error-code conventions.** Consider structured `errorCode` values for invalid handles, invalid options, context overflow, model load failures and inference failures.
- [ ] **Review Tcl object lifetimes around callbacks and strings.** Characterize callback reentrancy/error propagation and make sure Tcl-managed strings are retained for as long as native code may reference them.
- [ ] **Review multi-interpreter behavior.** Confirm that handles and any process-global llama.cpp initialization state behave correctly with more than one Tcl interpreter.

### llama.cpp compatibility and model characterization

- [x] **Pin and record a known-good llama.cpp baseline before modifying behavior.** Commit, compiler, backend and representative models are recorded in `BASELINE.md`.
- [x] **Revalidate current APIs against modern llama.cpp.** The binding is adapted and tested against the pinned `1719dbb0` revision; future updates still require a new review.
- [x] **Characterize chat templates and stop behavior across representative GGUF families.** Llama 3.2 1B and Qwen3 1.7B are covered; this does not imply universal template coverage.
- [x] **Test textual stop handling and UTF-8 boundaries.** Regression tests cover streaming and UTF-8 buffering on the two recorded models; broader model coverage remains open.
- [ ] **Characterize context overflow behavior.** Test prompts near `n_ctx`, continuation across calls, reset behavior and token accounting.

### Tests and evidence

- [x] **Add lifecycle regression tests.** Cover repeated init/free, command deletion, interpreter teardown, stale/foreign handles and multiple handles.
- [x] **Add parser regression tests.** Exercise supported options, invalid/missing/duplicate values and unknown options.
- [x] **Add real-model integration tests.** Verify tokenization, generation, chat templates, callbacks, stop IDs, cache/reset behavior and telemetry on a reproducible model set.
- [ ] **Run memory and undefined-behavior tooling.** Use ASan/UBSan and/or Valgrind where practical; retain the exact build/runtime evidence rather than documenting memory safety from inspection alone.
- [ ] **Add reproducible characterization records.** Record host, backend, model checksum, llama.cpp commit, tclllama commit, parameters and exact output for behavior/performance claims.

### Project documentation

- [x] **Establish `DECISIONS.md` and `CHARACTERIZATION.md` before behavioral changes that require new contract decisions or measurements.** API decisions, unresolved questions and measured evidence are recorded separately.
- [x] **Review broad compatibility claims.** Current docs scope tested models and backends to evidence in the selected baseline.
- [x] **Update examples only after the contract is settled.** README/API examples now use the current handle-based contract; the superseded build/API material is identified as historical.

---

*Review additions recorded: 2026-09-30.*
