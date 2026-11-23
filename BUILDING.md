# Building and testing tclllama

## Verified build configuration

The verified Linux x86_64 builds use llama.cpp commit
`1719dbb0ee7bdaf56de083079d3613bb670160a5` and its static
position-independent libraries. `Makefile.local` compiles the binding as
C++11, compiles the selected llama.cpp `common` sources and JSON Schema adapter
as C++17, and links `build/local/tclllama.so`.

The adapter uses the original `common/json`, `json-schema`,
`json-schema-to-grammar`, `unicode`, `common` and `trie` sources from that
checkout; they are compiled with PIC. Hidden visibility, function/data
sections and the GNU linker's `--gc-sections` retain only the needed utilities
from those common objects. The converter is not copied or reimplemented.

The makefile defaults point to the prepared sibling dependency checkout and
Tcl headers used for this build. For another directory layout, override them:

```sh
make -f Makefile.local -j2 \
    LLAMA_DIR=/path/to/llama.cpp \
    LLAMA_BUILD=/path/to/llama.cpp/build-static-pic \
    TCL_INCLUDE=/path/to/tcl/include
```

The Tcl headers must match Tcl 8.6 or later. The tested environment used Tcl
8.6.14 and GCC 15.3.1. The llama.cpp build used Release mode, `GGML_NATIVE=ON`,
OpenMP and PIC with static libraries.

The output includes a package index for the current internal package number
`7.5`. That number tracks the historical code progression; it is not the
version of the full library. The consolidated public release target is
`1.0.0`. The runtime/package metadata and the target version still need to be
reconciled before release.

## TEA configuration, installation and load check

The TEA configure flow has been verified on Fedora 43 with ActiveTcl 8.6.14
and the pinned llama.cpp static PIC build. It installs a versioned package
directory, a generated `pkgIndex.tcl`, the shared extension and the Tcl library
scripts. Build in a directory separate from the source checkout and select the
Tcl config, llama.cpp source tree and existing llama.cpp build explicitly:

```sh
mkdir -p build/tea
cd build/tea
../../configure \
    --prefix="$HOME/.local" \
    --with-tcl=/opt/ActiveTcl-8.6/lib \
    --with-llama=/path/to/llama.cpp \
    --with-llama-build=/path/to/llama.cpp/build-1719dbb-static-pic
make -j2
make install
```

The Tcl package version remains the internal version `7.5`; the public release
target is `1.0.0`. From any working directory, discover it with `TCLLIBPATH`
or add the package directory to `auto_path`:

```sh
TCLLIBPATH="$HOME/.local/lib/tclllama7.5" tclsh <<'TCL'
package require tclllama
puts [llama::engine_info]
TCL
```

Run `make uninstall` from the configured build directory to remove only the
package-owned module, index and Tcl scripts. The target removes the package
directory only when empty, preserving unrelated files. Re-running `make
install` replaces the package files in the same prefix.

The module statically includes the pinned llama.cpp/ggml archives and the
selected `common` support objects. Its Linux runtime dependencies include
OpenSSL `libcrypto`, the C/C++ runtimes, OpenMP and standard system libraries;
these are not copied into the prefix. No checkout or build-directory RPATH is
embedded. This is a Fedora/ActiveTcl validation, not a cross-platform package
or universal binary claim. The llama.cpp archives must be PIC-enabled and
compatible with the target CPU. Use `--with-tcl` to point at a directory
containing `tclConfig.sh` when Tcl is outside standard system paths.

## Local load check

```sh
TCLLIBPATH="$PWD/build/local" tclsh <<'TCL'
package require tclllama 7.5
puts [llama::version]
TCL
```

## Tests

The Tcl regression suites use an existing GGUF path. No model is downloaded by
the tests. `TCLLLAMA_TEST_CHAT=1` enables chat-specific checks for models with a
supported chat template.

```sh
for model in llama qwen; do
    case "$model" in
        llama) model_path=/path/to/llama3.2-1b.gguf ;;
        qwen)  model_path=/path/to/qwen3-1.7b.gguf ;;
    esac
    for suite in tests/*.test; do
        TCLLLAMA_TEST_MODEL="$model_path" TCLLLAMA_TEST_CHAT=1 \
            tclsh "$suite" build/local || exit 1
    done
done
```

The current Tcl suites cover arguments, compatibility, context continuity,
GBNF, lifecycle, logit bias, sampling, JSON Schema, native context/sequence
state, snapshots, strategies, streaming and shared native models with
independent contexts. The test names and test files are the source of truth for
coverage; passing these suites does not establish universal model
compatibility.

`tests/embeddings.test` uses `TCLLLAMA_TEST_EMBEDDING_MODEL` for its encoder or
non-causal embedding-model cases and `TCLLLAMA_TEST_DECODER_MODEL` for its
optional causal-decoder path. General model regression runs leave these
variables unset, so embedding-only cases are reported as skipped rather than
run against unrelated generative fixtures.

`tests/embeddings_batch.test` exercises native batching and Float32 output with
the same embedding-model variable; set `TCLLLAMA_TEST_DECODER_MODEL` to run its
causal decoder batch case. The repeatable CPU benchmark writes TSV to stdout:

```sh
tclsh tests/embeddings_benchmark.tcl build/local /path/to/model.gguf model-label 7
tclsh tests/embeddings_benchmark.tcl build/local /path/to/nomic.gguf nomic 7 'search_document: '
```

Use at least five repetitions. The benchmark separates model load, context
creation, token-count probing, first inference and warm end-to-end operations;
it records model hash, effective capacities, median/MAD, throughput, native
call count by API contract, and Linux RSS/HWM when `/proc` is available. It
does not split tokenization, native evaluation and extraction inside an
inference operation, and its CPU results are not portable performance claims.
Pass an optional sixth argument to select the context thread count:

```sh
tclsh tests/embeddings_benchmark.tcl build/local /path/to/model.gguf minilm 7 {} 4
```

## Reusable host/lab runner

`scripts/lab/inspect.tcl` reports OS, CPU flags (including individual
AVX/AVX-512 subsets), RAM/swap, compiler, Tcl, optional GPU diagnostic tools,
and llama.cpp backend/device information when given a package directory.
Unavailable tools are marked `unsupported`; missing GPU diagnostics do not
stop the report. It detects advertised capabilities only and does not prove
that a backend can load or execute a model.

`scripts/lab/run.tcl` reuses the existing test suites, native checks, embedding
benchmark, and a short generation/chat/shared-round benchmark. Results and
logs default outside the Git repository under `../experimentos/`. Benchmark
stdout is TSV and diagnostics are saved in a separate `.stderr` file. In the
generation benchmark, `units` is the median number of decoded tokens for
generation and chat, `requested_tokens` records the limit, and throughput is
the median decoded count divided by the median end-to-end time. Other rows
retain their existing units and report `NA` for the requested-token field.
Example for a fresh CPU host, after checking the flags reported by `inspect.tcl`:

```sh
tclsh scripts/lab/inspect.tcl -output ../experimentos/laboratorio-055/environment.tsv
tclsh scripts/lab/run.tcl -scenarios engine-build \
    -llama-build-dir build/llama-lab -cc gcc -cxx g++ \
    -engine-options '-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON'
tclsh scripts/lab/run.tcl -scenarios 'build regression testing native sanitizers benchmark' \
    -build-dir build/lab-cpu -test-build-dir build/lab-testing \
    -llama-build-dir build/llama-lab \
    -cxx g++ -cxxflags '-O2' -threads 4 -repetitions 5 \
    -results ../experimentos/laboratorio-055/cpu.tsv
```

The engine-build scenario explicitly disables native CPU ISA flags by default,
because this llama.cpp revision enables several per-ISA CMake options by
default even with `GGML_NATIVE=OFF`. Add only flags verified on the target host
through `-engine-options`; the example is valid only if AVX, AVX2, FMA and
F16C are all advertised. It produces a host-specific binary, not a generic
binary for Gepeto. The AVX-512 lab must select only the subsets reported by
that host; an available `GGML_AVX512` option does not itself verify execution.

The runner accepts explicit repository, model-store, Tcl, compiler, build,
model, scenario, repetition, thread and result paths. It resolves the four
local GGUF fixtures by their recorded SHA-256 blob names by default; it never
downloads them. Missing fixtures or GPU utilities are not silently counted as
passes. The result TSV uses `pass`, `fail`, `skip`, `unsupported` and
`not_run` statuses, and full Tcltest output is stored in adjacent log files.
The native CPU runner does not claim AVX/AVX2, AVX-512, or GPU coverage merely
because a later host reports those features.

The generation baseline reports load, context creation, prefill, 16-token
sequence generation, chat, and one two-sequence `batch_step`. The embedding
baseline reuses `tests/embeddings_benchmark.tcl` for individual versus native
batch calls, including Float32 output and context/model metadata. These are
bounded characterization runs, not cross-host performance guarantees.

The laboratory procedure separates the upcoming hosts:

- **CPU AVX/AVX2:** inspect the actual instruction flags first, choose only a
  compatible CMake/ggml configuration (the example is valid only when AVX2 is
  advertised), build in a fresh directory, then run regression,
  testing hooks, native checks, sanitizers, and bounded benchmarks. A machine
  advertised as AVX2 is not presumed to have a GPU.
- **AVX-512:** run only if the CPU advertises the required subsets; do not reuse
  a binary built with unsupported subsets. **GPU** is a separate branch that
  requires a detected device, working driver, compiled backend, and an actual
  successful model load/offload check. AVX-512 does not imply GPU availability.

For each lab, preserve `environment.tsv`, the runner TSV, and per-suite logs
outside Git. Compare identical GGUF hashes, prompts, context/batch settings,
thread counts, warm-up policy, and repetitions. Record unavailable GPU/ISA tests
as `unsupported` or `not_run`, with their reason; neither status is a failure or
a pass. Gepeto is the pre-lab CPU baseline and does not execute remote tests.

`tests/shared_model.test` runs against both the normal and `TCLLLAMA_TESTING`
builds. The testing build exposes a private pointer-identity probe so the suite
can distinguish contexts using the same native `llama_model` from two models
loaded from the same GGUF path. That command is absent from normal package
builds.

Model-free native checks:

```sh
make -f Makefile.local sampler-history-test logit-bias-test \
    sampling-strategies-test schema-adapter-test
make -f Makefile.local gguf-fixture
c++ -std=c++11 tests/stream_buffer.cpp -o /tmp/tclllama-stream-buffer-test
tclsh tests/gguf.test build/local
/tmp/tclllama-stream-buffer-test
```

`gguf-fixture` builds a small metadata-only GGUF through the public gguf.h
setters. `tests/gguf.test` can run without inference weights for its synthetic
cases. To enable its real-model inspection case, set
`TCLLLAMA_TEST_MODEL=/path/to/model.gguf` when invoking `tclsh`.

`schema.test` uses Python 3 to parse generated JSON and check the specific
properties exercised by those cases. It is not a general JSON Schema
validator. The current test setup does not verify GPU execution.

## Build and distribution limits

- Only the Linux CPU build described above has been verified in this baseline.
- GPU options are exposed by the API, but GPU support has not been tested on
  GPU hardware.
- `Makefile.local` and the TEA installation expect PIC-enabled llama.cpp static
  libraries. Distribution of those dependencies and cross-platform packaging
  remain outside the verified procedure.

## AddressSanitizer and LeakSanitizer on Linux

The audit build was verified on Gepeto with GCC 15.3.1, Tcl 8.6.14 and
Fedora's `libasan` runtime. Install `libasan` if the sanitizer link step cannot
find the runtime. Leak detection is enabled through ASan with
`detect_leaks=1`; a separate `liblsan` runtime was not required by this build.

```sh
make -f Makefile.local -j2 \
    BUILD_DIR=build/audit-asan \
    CXX='g++ -fsanitize=address' \
    CXXFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address' all

LD_PRELOAD=/usr/lib64/libasan.so.8 \
ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:alloc_dealloc_mismatch=1:strict_string_checks=1' \
TCLLIBPATH="$PWD/build/audit-asan" \
TCLLLAMA_TEST_MODEL=/path/to/model.gguf \
tclsh tests/lifecycle.test build/audit-asan
```

For the private fault-injection suite, use a separate build directory and add
`-DTCLLLAMA_TESTING` to `CXXFLAGS`, then run `tests/failure_injection.test`
against that package. Do not set the test define in a normal build. Preloading
the ASan runtime is necessary when loading the extension into an uninstrumented
`tclsh` executable.

This procedure instruments `generic/tclllama.c`, the JSON Schema adapter and
the `common/*.cpp` objects compiled by this Makefile. The prebuilt static
llama.cpp/ggml archives and Tcl runtime remain uninstrumented; sanitizer
results cover the instrumented binding paths and boundary calls, not every
native dependency. Keep sanitizer artifacts separate from normal builds and
do not compare their RSS measurements with a normal build.

For adapter lifecycle tests, generate a tensor-free upstream GGUF fixture:

```sh
make -f Makefile.local BUILD_DIR=build/local lora-fixture
build/local/lora-fixture llama /tmp/empty-llama-adapter.gguf
TCLLLAMA_TEST_LORA=/tmp/empty-llama-adapter.gguf \
TCLLLAMA_TEST_MODEL=/path/to/llama-model.gguf \
  /opt/ActiveTcl-8.6/bin/tclsh tests/model_controls.test build/local
```

Use `qwen3` instead of `llama` for the local Qwen model. This fixture contains
metadata and no adapter tensors; it tests native loading, attachment and
ownership, not learned weight adaptation. It must not be presented as a real
LoRA inference fixture. The fixture is generated locally, not downloaded.

`tests/model_loading.test` exercises native device arrays and typed overrides
with either local GGUF. Run it with TCLLLAMA_TEST_MODEL as for other suites.
Directed sanitizer runs use the isolated ASan build and libasan preload already
described above; no GPU runtime or extra fixtures are needed. CPU tests do not
validate CUDA/Vulkan/ROCm distribution or nondefault direct-I/O/mlock modes.
