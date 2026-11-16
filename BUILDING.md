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
GBNF, lifecycle, logit bias, sampling, JSON Schema, snapshots, strategies and
streaming. The test names and test files are the source of truth for coverage;
passing these suites does not establish universal model compatibility.

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
