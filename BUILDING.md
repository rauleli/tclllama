# Building and testing tclllama

## Verified build configuration

The currently verified build is a local Linux x86_64 build against llama.cpp
commit `1719dbb0ee7bdaf56de083079d3613bb670160a5`, using its static
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
OpenMP and PIC with static libraries. This local recipe is not a portable TEA
packaging, installation or cross-platform build procedure. The older
`configure.ac`/TEA path has not been revalidated against this implementation.

The output includes a package index for the current internal package number
`7.5`. That number tracks the historical code progression; it is not the
version of the full library. The consolidated public release target is
`1.0.0`. The runtime/package metadata and the target version still need to be
reconciled before release.

## Load check

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
GBNF, lifecycle, logit bias, sampling, JSON Schema, snapshots, strategies,
streaming and shared native models with independent contexts. The test names
and test files are the source of truth for coverage; passing these suites does
not establish universal model compatibility.

`tests/shared_model.test` runs against both the normal and `TCLLLAMA_TESTING`
builds. The testing build exposes a private pointer-identity probe so the suite
can distinguish contexts using the same native `llama_model` from two models
loaded from the same GGUF path. That command is absent from normal package
builds.

Model-free native checks:

```sh
make -f Makefile.local sampler-history-test logit-bias-test \
    sampling-strategies-test schema-adapter-test
c++ -std=c++11 tests/stream_buffer.cpp -o /tmp/tclllama-stream-buffer-test
/tmp/tclllama-stream-buffer-test
```

`schema.test` uses Python 3 to parse generated JSON and check the specific
properties exercised by those cases. It is not a general JSON Schema
validator. The current test setup does not verify GPU execution.

## Build and distribution limits

- Only the Linux CPU build described above has been verified in this baseline.
- GPU options are exposed by the API, but GPU support has not been tested on
  GPU hardware.
- The current `Makefile.local` expects PIC-enabled llama.cpp static libraries.
  The final library distribution and dependency-link strategy remain open.
- No install target or cross-platform packaging claim is made here.

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
