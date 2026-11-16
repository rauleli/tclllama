# Technical baseline

## Source revisions

- Project: `tclllama`, repository checkout `base_publicada/`, branch `main`.
- Functional consolidation: `4f287d9c8c52f13e0777225cfd67d1f0cbf09397`.
- Local merge of `origin/main`: `222fd3e` (the remote-side changes were confined to
  the public September `TODO.md`; the merge did not alter binding code).
- llama.cpp: `1719dbb0ee7bdaf56de083079d3613bb670160a5` in the sibling checkout
  `../dependencias/llama.cpp`.

The llama.cpp build used for this baseline has `BUILD_SHARED_LIBS=OFF`, Release
mode, `CMAKE_POSITION_INDEPENDENT_CODE=ON`, `GGML_NATIVE=ON` and OpenMP enabled.
The extension links its static PIC libraries. The exact local build instructions
are in [BUILDING.md](BUILDING.md).

## Toolchain and backend

- Linux x86_64; AMD Phenom II X4 965 CPU.
- Tcl 8.6.14; GCC 15.3.1.
- CPU inference backend; `GGML_NATIVE=ON` and OpenMP.
- No GPU verification is included in this baseline.

The working build is `Makefile.local`, and the verified output is
`build/local/tclllama.so`. This does not establish a portable TEA package,
installation procedure, shared-library strategy or support for other operating
systems.

## Model fixtures

The local GGUF blobs are stored outside Git under `../datos/modelos/blobs/`.
Their SHA-256 digests were checked against their filenames:

| Model | Quantization | SHA-256 | Size |
| --- | --- | --- | ---: |
| Llama 3.2 1B Instruct | Q8_0 | `74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` | 1,321,082,688 bytes |
| Qwen3 1.7B | Q4_K_M | `3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6` | 1,359,279,776 bytes |

The tests point at existing files and do not download models.

## Current package labels

The build/package metadata and `llama::version` currently report `7.5`, a
historical internal code/package progression. The planned consolidated public
release is `1.0.0`. The runtime metadata, package requirements and distribution
version still need an explicit release decision and reconciliation.

## Memory diagnostics available on Gepeto

The lifecycle characterization used GCC 15.3.1, Fedora `libasan` 15.3.1,
Valgrind 3.27.1 and Tcl 8.6.14. Leak detection was enabled through ASan; a
separate `liblsan` package was not required. The Tcl measurement harness reads
Linux `/proc/self/status`, `smaps_rollup` and `smaps`; these measurements are
Linux-specific and are not part of the binding API.
