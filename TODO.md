# TODO — tclllama revalidation and hardening

This file records follow-up work identified during a repository review on **2026-09-30**.

The intent is not to rewrite the project indiscriminately. The current implementation already contains substantial functionality; the next pass should establish the real contract, verify lifecycle and state semantics, and then evolve the binding in small evidence-backed slices.

## Review follow-up — 2026-09-30

### API and documentation reconciliation

- [ ] **Reconcile the public API with the implementation.** The current C implementation is handle-based (`llama::init` returns a handle consumed by `llama::generate`, `llama::chat`, `llama::free`, etc.), while parts of README/API documentation still show singleton-style calls. Decide the intended contract first, then make code, examples, package/version text and documentation agree.
- [ ] **Reconcile project/package versioning.** Review the repository's `v1.0` presentation versus `Tcl_PkgProvide(..., "7.5")` / Ik'nal v7.5 naming, and define what each version number means.
- [ ] **Audit documented options against actual parsing and upstream llama.cpp.** Remove or correct options that are no longer implemented, have changed upstream, or have different defaults/semantics.
- [ ] **Separate contractual behavior from observed/model-specific behavior.** Avoid documenting a behavior as universal until it has been characterized across the intended model families and llama.cpp baseline.

### Handle identity, ownership and lifecycle

- [ ] **Stop relying on pointer-derived handle names as durable identity.** Current handles are generated with `"llama%p"`. Revisit identity using an interpreter-local monotonic scheme or another collision-resistant mechanism, informed by the handle-reuse issue already characterized in tclwhisper.
- [ ] **Give handle commands an explicit delete callback.** Make explicit `llama::free`, `rename $handle {}`, command deletion and interpreter destruction converge on one cleanup path.
- [ ] **Make cleanup idempotent and ownership explicit.** Define and test ownership/order for sampler, context, model and any future per-handle allocations.
- [ ] **Strengthen handle validation.** Do not accept an arbitrary Tcl command merely because `Tcl_GetCommandInfo` succeeds. Verify that the command was created by tclllama and that its client data and command identity are internally consistent.
- [ ] **Test stale, renamed, deleted and foreign-command handles.** Invalid handles must fail deterministically without dereferencing unrelated `ClientData`.

### State and option semantics

- [ ] **Decide which generation options are per-call and which belong to the handle.** `apply_options()` currently mutates persistent `LlamaState`; determine whether values such as temperature, sampling parameters, seed and token limits should persist into later calls or revert to defaults when omitted.
- [ ] **Define `generate` versus `chat` state semantics.** Document and test KV-cache persistence, `-reset`, chat's unconditional reset, system-message insertion, and transitions between `generate` and `chat` on the same handle.
- [ ] **Prevent silent state leakage between calls.** Once the intended semantics are chosen, add regression tests proving omitted options and previous calls cannot change later behavior unintentionally.
- [ ] **Review sampler rebuild semantics.** Verify the order and compatibility of temperature, top-k, top-p, min-p, penalties, Mirostat and distribution sampling against the selected llama.cpp version.

### Argument parsing and Tcl behavior

- [ ] **Harden option parsing.** Missing option values, unknown options, duplicate options and failed Tcl conversions should produce deterministic Tcl errors rather than being silently ignored or partially applied.
- [ ] **Define error-code conventions.** Consider structured `errorCode` values for invalid handles, invalid options, context overflow, model load failures and inference failures.
- [ ] **Review Tcl object lifetimes around callbacks and strings.** Characterize callback reentrancy/error propagation and make sure Tcl-managed strings are retained for as long as native code may reference them.
- [ ] **Review multi-interpreter behavior.** Confirm that handles and any process-global llama.cpp initialization state behave correctly with more than one Tcl interpreter.

### llama.cpp compatibility and model characterization

- [ ] **Pin and record a known-good llama.cpp baseline before modifying behavior.** Capture upstream commit/version, compiler, backend and representative models.
- [ ] **Revalidate current APIs against modern llama.cpp.** The upstream library changes quickly; identify API drift before deciding what is a tclllama bug versus an obsolete upstream assumption.
- [ ] **Characterize chat templates and stop behavior across representative GGUF families.** Include at least models using different chat templates/control-token conventions rather than assuming one universal path.
- [ ] **Test textual stop handling and UTF-8 boundaries.** Preserve complete UTF-8 sequences and verify that control-token/text-buffer logic neither leaks template markers nor truncates valid output.
- [ ] **Characterize context overflow behavior.** Test prompts near `n_ctx`, continuation across calls, reset behavior and token accounting.

### Tests and evidence

- [ ] **Add lifecycle regression tests.** Cover repeated init/free, implicit deletion, interpreter teardown, invalid/double free, multiple simultaneous handles and handle-name reuse attempts.
- [ ] **Add parser regression tests.** Exercise every supported option, invalid values, missing values, duplicate values and unknown options.
- [ ] **Add real-model integration tests.** Verify tokenize/detokenize, generation, chat templates, callbacks, stop IDs, cache/reset behavior and metrics on a small reproducible model set.
- [ ] **Run memory and undefined-behavior tooling.** Use ASan/UBSan and/or Valgrind where practical; retain the exact build/runtime evidence rather than documenting memory safety from inspection alone.
- [ ] **Add reproducible characterization records.** Record host, backend, model checksum, llama.cpp commit, tclllama commit, parameters and exact output for behavior/performance claims.

### Project documentation

- [ ] **Introduce decision/characterization documents if useful.** A lightweight `DECISIONS.md` and `CHARACTERIZATION.md` can separate API contracts from measurements and unresolved questions, following the method now used in tclwhisper.
- [ ] **Review broad compatibility claims.** Statements such as universal GGUF/model support should be scoped to what is actually exercised and supported by the selected llama.cpp baseline.
- [ ] **Update examples only after the contract is settled.** Preserve historical context where useful, but make the main README/API examples executable against the current implementation.

---

*Review additions recorded: 2026-09-30.*
