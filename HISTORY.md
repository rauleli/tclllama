# Technical history

This is a concise history of implementation and evidence, not a record of
private conversations or personal background.

## Repository and version provenance

The inherited repository contains a historical `1.0` changelog entry dated
2024-12-21 and code that reports package progression `7.5`. The latter
describes an internal code iteration, not the version of the whole library.
The intended public release for the consolidated project is `1.0.0`; package
metadata reconciliation is still pending.

The functional work from the documentary audits was consolidated locally in
commit `4f287d9` rather than represented as one commit per audit. The later
merge commit `222fd3e` integrated two remote documentation commits containing
the September revalidation TODO. That merge changed no functional code.

## Implementation milestones

- **Audits 001–004 — recovery and llama.cpp adaptation:** identified the local
  repository and implementation, pinned llama.cpp to `1719dbb0`, adapted the
  binding to that API, built it and began real-model verification.
- **Audits 005–011 — stable core behavior:** consolidated handle ownership and
  cleanup, hardened Tcl argument parsing, characterized context continuation
  and sampler history, corrected streaming/termination behavior, then inventoried
  capabilities and remaining work.
- **Audits 012–013 — configuration and model inspection:** added validated
  model/context initialization parameters and expanded `llama::info` with GGUF
  and runtime metadata.
- **Audits 014–015 — snapshots:** evaluated native context-state APIs and added
  one temporary in-memory snapshot per handle, including context, sampler and
  persistent sampling parameters.
- **Audits 016–018 — GBNF:** defined the generation-finalization contract and
  integrated GBNF through the native sampler path, with candidate validation,
  streaming treatment and snapshot support.
- **Audit 019 — JSON Schema:** integrated llama.cpp's native JSON Schema to GBNF
  converter with a bounded, explicitly validated input subset.
- **Audit 020 — logit bias:** added token-ID adjustments and suppression with
  validation and guarded sampling.
- **Audit 021 — sampling strategies:** added native Typical, DRY and XTC options,
  with explicit rejection of incompatible Mirostat combinations.
- **Audit 022 — local consolidation:** created the functional consolidation
  commit after rerunning model and native regressions.

Detailed per-iteration evidence is retained outside Git in the private audit
files. This repository records the resulting technical state, not copies of
those reports.

## Inherited README material reconciled

The previous README included a TEA/shared-library recipe: install Tcl and
OpenSSL development files plus a C++/Autoconf toolchain; build llama.cpp with
`BUILD_SHARED_LIBS=ON`; run `autoconf`, configure with
`./configure --with-llama=/path/to/llama.cpp`, then `make` and optionally
`make install`. It described searching `include/`, `ggml/include/` and several
shared-library locations. These are retained here as the historical procedure
that was documented, not evidence that it builds the consolidated binding.
`BUILDING.md` records the currently verified local static-PIC build instead.

The inherited README also contained singleton-style commands (`llama init`,
`llama generate`, `llama free`), outdated `llama::info` keys, an unregistered
`getcpuinfo` command, fake/example token IDs, an incorrect registry namespace,
and unsupported claims of universal GGUF and cross-platform support. The
current workflows were corrected or moved to `API.md`; invalid examples and
unsupported claims were removed. Its interactive chat, sequential prompt-file
processing, model inspection, resource guidance, troubleshooting, architecture
and project-tree sections were retained in corrected form in the current
README. The old creator-support solicitation was omitted because it is not
technical project documentation.

### Content classification of the superseded README

| Previous material | Classification | Reconciliation |
| --- | --- | --- |
| Namespaced/native Tcl use, generation, tokenization, telemetry, model reuse, context sizing, prompt-file workflow and support/contribution links | Still useful | Kept, corrected to current commands, or linked to the current API. The chat, sequential prompt processing and model inspection examples remain in README. |
| `llama init/generate/free`, singleton state, old `llama info` keys, invented token examples, multi-level `verbose` and `getcpuinfo` | Incorrect/obsolete for this checkout | Replaced by handle-based `llama::` examples. `getcpuinfo` was omitted because the current binding does not register it. The old forms remain recoverable from the prior README blob in Git history. |
| “all GGUF”, universal model support, cross-platform support, “llama.cpp latest”, and old current-version claims | Unsupported or stale | Removed as current claims. Baseline and characterization records scope verified configurations. The older `v1.0` label is retained as historical repository documentation, not as current release metadata. |
| Detailed old option tables | Superseded and partly invalid | Current ranges/defaults and accepted options are in `API.md`; keys not accepted by this parser were not carried forward. |
| TEA/shared-library build procedure | Historical, not revalidated | The previous README prescribed the following process; it is recorded here to preserve provenance, not as build advice: |
| Performance suggestions, troubleshooting, architecture and source-tree overview | Useful with correction | Retained in README with qualifications, current command names and the present repository layout. |
| Creator-support section | Nontechnical project promotion | Its support link remains in README; promotional explanatory text was shortened. |

The previous README also listed these OS package commands and build steps. They
are archived as documentation history only; the current supported local build
procedure is in `BUILDING.md`:

```sh
dnf install -y tcl-devel openssl-devel gcc-c++ cmake make autoconf automake libtool git
apt-get install -y tcl-dev libssl-dev build-essential cmake make autoconf automake libtool git
git clone https://github.com/ggerganov/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON
cmake --build llama.cpp/build --config Release
autoconf
./configure --with-llama=/path/to/llama.cpp
make
make install
```

That historical recipe targeted shared libraries and an older TEA flow. It has
not been demonstrated to build this consolidated source against the pinned
llama.cpp revision.

The former `BUILDING.md` additionally contained unverified platform recipes for
Fedora/Debian packages, macOS and Windows, optional CUDA/HIP/Metal builds,
shared linking, installation, compiler flags and troubleshooting. These were
not retained as current procedures because this baseline verifies only the
Linux CPU static-PIC build. The previous files remain recoverable from Git
history. Numeric memory and token-speed estimates from the old API guide were
also excluded because they were not tied to a recorded model/build/hardware
measurement. The current docs retain qualitative resource guidance and the
actual test conditions instead.
