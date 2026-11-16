# tclllama — llama.cpp for Tcl

`tclllama` is a native Tcl extension around **llama.cpp**. Its compatible
`llama::init` command creates a model and context together; the explicit
`llama::model_load` and `llama::context_create` commands can instead reuse one
loaded native model across independent context handles. The API provides
generation, chat, tokenization, model/context inspection, sampling controls and
temporary in-memory inference snapshots. The selected llama.cpp revision and
backend determine which models can run; this project does not claim that every
GGUF architecture or backend has been tested.

## Project vision

The goal of tclllama is to expose the functional capabilities of llama.cpp
systematically through a native Tcl binding. Tcl projects can use the binding
directly, without relying on Ollama or an external service. Complementary Tcl
utilities can add functionality, as the existing model manager does. A later,
optional module is planned to provide the Ollama-compatible services that can
be implemented with tclllama.

This vision extends beyond the capabilities implemented today, which are listed
in the following section. See [ROADMAP.md](ROADMAP.md) for the work ahead.

## Why Tcl?

It all started with Eggdrop bots in the '90s. Things got a little out of hand.

Thirty years later, I'm still using it to connect databases, automate industrial systems, build ERPs, run distributed workers, and integrate AI models.

Some of that code has been running unchanged for seven years.

At this point, switching languages would be the experimental choice.

## Documentation

- [API.md](API.md) — current commands and contracts.
- [BUILDING.md](BUILDING.md) — verified build and tests.
- [ROADMAP.md](ROADMAP.md) — pending capabilities and direction.
- [DECISIONS.md](DECISIONS.md) — architectural decisions and contracts.
- [CHARACTERIZATION.md](CHARACTERIZATION.md) — experimental evidence and limits.
- [HISTORY.md](HISTORY.md) — verified technical evolution.

## Features

- **Native Tcl API**: Use the binding directly without a required daemon or
  external inference process.
- **Shared model, separate contexts**: Load one native model object and create
  multiple independent contexts with their own KV and inference state. Calls
  using that model are serialized; this does not add concurrent inference.
- **Text and chat inference**: Continue or reset context, stream fragments to
  callbacks, limit output and stop on selected token IDs.
- **Sampling**: Temperature, top-k/top-p/min-p, Mirostat, logit bias, Typical,
  DRY and XTC, with validation and per-handle persistence.
- **Constrained generation**: Native GBNF and a documented JSON Schema subset.
- **Multi-sequence inference**: Submit prompt-token batches for distinct sequences
  through one native decode call, with per-sequence logits retained for later
  independent generation where enabled. Independent sequences support raw-token
  and text generation, including UTF-8 callbacks. They can also advance in shared
  decode rounds, and active prefixes can be forked with cloned sampler state.
  Temporary in-memory snapshots preserve the complete sequence table and
  sampler state for supported decoder models.
- **Model inspection**: Query GGUF metadata, effective context settings and
  inference telemetry.
- **Auxiliary model utility**: `library/ollama_registry.tcl` downloads,
  lists, inspects and deletes local models from the Ollama registry.

The current functional baseline has been exercised on Linux x86_64 with the
CPU backend and two model files. GPU, macOS and Windows execution have not been
verified in this project baseline.

## Prerequisites and build status

The verified local build uses Tcl 8.6 development headers, a C++ compiler and
the static PIC llama.cpp libraries from commit
`1719dbb0ee7bdaf56de083079d3613bb670160a5`. See
[BUILDING.md](BUILDING.md) for the exact local command and its limits.

### Legacy TEA build notes

The repository still contains `configure.ac`, `Makefile.in` and Tcl Extension
Architecture (TEA) support. The inherited instructions for shared-library
detection, OpenSSL and cross-platform builds are retained in the source
history, but that path has not been revalidated against the current binding and
llama.cpp API. Do not treat those legacy instructions as the verified build.

## Quick Start

### Loading the Extension

```tcl
package require tclllama 7.5
```

### Basic Example: Generate Text

```tcl
# Initialize a model
set model_path "/path/to/model.gguf"
set handle [llama::init $model_path 2048]

# Generate text
set response [llama::generate $handle "Hello, how are you?" -max_tokens 100]
puts $response

# Cleanup
llama::free $handle
```

### Advanced Example: Custom Sampling Parameters

```tcl
set handle [llama::init "model.gguf" 4096]

set options {
    temperature 0.7
    top_k 40
    top_p 0.95
    num_predict 256
}

set response [llama::generate $handle "Explain quantum computing in simple terms" \
    -options $options]
puts $response

llama::free $handle
```

## API Reference

The complete command contract, option ranges, defaults, errors and edge cases
are documented in [API.md](API.md). The current API uses namespaced commands
and an explicit handle:

```tcl
set handle [llama::init model.gguf 2048 \
    -options [dict create n_threads 4 n_batch 512]]
try {
    set text [llama::generate $handle "What is Tcl?" -max_tokens 80]
    puts $text
} finally {
    llama::free $handle
}
```

Sampling values are passed as a Tcl dictionary and persist on that handle.
Common defaults include temperature `0.8`, top-k `40`, top-p `0.95`, min-p
`0.05`, repeat penalty `1.1`, Mirostat disabled and `num_predict -1`. Use
`-max_tokens` for a single call; it cannot be combined with
`-options num_predict`.

Chat takes a list of message dictionaries. Each message needs a non-empty
`role` and a `content` value. A model chat template is used when present:

```tcl
set messages [list \
    [dict create role system content "Answer briefly."] \
    [dict create role user content "What is Tcl?"]]
set answer [llama::chat $handle $messages -max_tokens 80]
```

Each chat call resets inference state and formats the message list anew. Pass
conversation history in the list on each call when needed.

Tokenization uses the loaded model vocabulary; token IDs and decoded pieces
vary by model:

```tcl
set ids [llama::tokenize $handle "Hello world"]
set text [llama::detokenize $handle $ids]
```

`llama::info $handle` returns context settings, model metadata, sampling values
and telemetry. `llama::version` reports tclllama package/build metadata, not the
llama.cpp revision. `llama::verbose $handle ?boolean?` toggles per-handle token
diagnostics. `llama::clear_cache $handle` resets inference context and sampler
history. The full API reference also covers snapshots, constrained generation,
callbacks and stop behavior.

## Tcl Library: `ollama_registry.tcl`

The repository includes `library/ollama_registry.tcl`, a separate Tcl utility
for the Ollama model registry and local model manifests. Its current procedures
include `download_model`, `list_local_models`, `delete_model`, `get_model_path`,
`show_model_info` and `show_config`. It is not the native binding's model
loader; pass a resolved local GGUF path to `llama::init`.

```tcl
source library/ollama_registry.tcl

set local_models [ollama_registry::list_local_models]
foreach model $local_models {
    puts "Model: $model"
}

# Resolve a GGUF path only when a local entry exists.
if {[llength $local_models]} {
    set model_path [ollama_registry::get_model_path [lindex $local_models 0]]
}
```

The manager's public procedures are implemented in the Tcl source but are not
yet covered by a dedicated registry regression suite. Review the procedure
contracts in that file before using download or deletion operations.

## Performance and resource notes

- **Reuse a loaded handle** for several operations instead of repeatedly loading
  the model. Free it when finished.
- **Choose context size for the workload.** A larger context may require more
  memory; it does not guarantee better output and may increase prompt cost.
- **Bound generated output** with `-max_tokens`. `num_predict` is a persistent
  sampling option; see [API.md](API.md) for its interaction with other options.
- Temperature and filtering settings change sampling behavior, but their
  quality effects depend on the model and prompt. The project does not provide
  universal tuning values or comparative benchmark claims.
- As a general sampling tendency, lower temperature is less variable and
  higher temperature is more variable; this is not a guarantee of quality or
  determinism across models.
- `llama::batch_prefill` can decode prompt tokens from distinct sequences in
  one call. `llama::batch_step` can then sample one token per selected sequence
  and decode their ordinary tokens together. These synchronous rounds are not
  continuous batching or concurrent execution.

## Examples

### Interactive Chat

```tcl
#!/usr/bin/env tclsh

package require tclllama 7.5

set handle [llama::init "model.gguf" 2048]
set messages {}
try {
    while {1} {
        puts -nonewline "> "
        flush stdout
        if {[gets stdin user_input] < 0 || $user_input eq "exit"} {
            break
        }

        lappend messages [dict create role user content $user_input]
        set response [llama::chat $handle $messages -max_tokens 200]
        puts "AI: $response\n"
        lappend messages [dict create role assistant content $response]
    }
} finally {
    llama::free $handle
}
```

### Batch Processing

```tcl
#!/usr/bin/env tclsh

package require tclllama 7.5

set input_file [open "prompts.txt" r]
set output_file [open "results.txt" w]
set handle [llama::init "model.gguf" 1024]
try {
    while {[gets $input_file line] >= 0} {
        if {[string trim $line] eq ""} {
            continue
        }
        set result [llama::generate $handle $line -reset 1 -max_tokens 150]
        puts $output_file "Input: $line"
        puts $output_file "Output: $result"
        puts $output_file "---"
    }
} finally {
    close $input_file
    close $output_file
    llama::free $handle
}
```

### Shared-Round Sequence Generation

For synchronous generation in a shared context, register and prefill multiple
sequences, then advance them one token at a time with `llama::batch_step`:

```tcl
set handle [llama::init "model.gguf" 1024 -options {
    n_seq_max 2 sequence_sampling true n_batch 32
}]
try {
    llama::sequence $handle create 1
    set prompt [llama::tokenize $handle "Hello"]
    llama::batch_prefill $handle [list [list 0 $prompt] [list 1 $prompt]]

    set step [llama::batch_step $handle {0 1}]
    puts [dict get $step sequences]
} finally {
    llama::free $handle
}
```

Each sequence retains its own sampler and logits. A round uses one public
`llama_decode` call for the ordinary tokens selected in that round. Applications
drive subsequent rounds themselves; tclllama does not schedule requests or run
generation asynchronously.

### Model Information Query

```tcl
package require tclllama 7.5

set handle [llama::init "model.gguf"]
try {
    set info [llama::info $handle]
    puts "Context size: [dict get $info n_ctx]"
    puts "Vocabulary size: [dict get $info n_vocab]"
    puts "Architecture: [dict get $info model architecture]"

    # This reports tclllama package/build metadata, not llama.cpp's revision.
    puts [llama::version]
} finally {
    llama::free $handle
}
```

## Troubleshooting

### Build cannot find llama.cpp or Tcl headers

Use the `LLAMA_DIR`, `LLAMA_BUILD` and `TCL_INCLUDE` overrides documented in
[BUILDING.md](BUILDING.md). A missing shared library concerns the legacy shared
build path, which is not the current verified build.

### Model load or generation fails

Check that the model path exists and points to a GGUF readable by the selected
llama.cpp build. Record the model, llama.cpp revision, backend and exact Tcl
error when reporting a failure. `llama::verbose $handle 1` enables the
binding's per-token diagnostic output; it is not a general llama.cpp log-level
control.

The binding reports Tcl errors for invalid options, handles and several native
failures. Do not treat a generic memory or segmentation-fault suggestion as a
diagnosis; reproduce the case with the test model/build and inspect native
diagnostics.

### Slow Performance

**Solution:**
- Reduce context size or output limits only if appropriate for the workload.
- Sampling changes are not guaranteed to improve speed; compare under the same
  model and build if performance is the concern.
- The prepared CPU build enables `GGML_NATIVE` and OpenMP. Other backend and
  hardware performance has not been established by this README.

## Architecture

```
┌─────────────────────────────────────┐
│   Tcl Scripts (.tcl)                │
├─────────────────────────────────────┤
│   tclllama Tcl extension             │
│   ├─ Handle-based model/context API  │
│   ├─ Generation, chat and sampling   │
│   └─ Metadata and in-memory snapshots│
├─────────────────────────────────────┤
│   llama.cpp (selected local build)  │
│   ├─ GGML Operations               │
│   ├─ Sampling & Inference          │
│   └─ Quantization Support          │
├─────────────────────────────────────┤
│   Hardware                          │
│   ├─ CPU (verified in this baseline)│
│   └─ GPU (API options; unverified)  │
└─────────────────────────────────────┘
```

## Project Structure

```
tclllama/
├── generic/
│   ├── tclllama.c           # Tcl binding and native inference
│   ├── sampling_guard.h     # Candidate validation helpers
│   ├── schema_adapter.cpp   # llama.cpp JSON Schema bridge
│   ├── stream_buffer.h      # UTF-8 and streaming buffer
│   └── sha256.c             # Existing TEA source; not used by local recipe
├── library/
│   └── ollama_registry.tcl  # Optional model registry utility
├── tests/                   # Tcl and native regression tests
├── tclconfig/               # Existing TEA configuration macros
├── configure.ac             # Legacy TEA configuration input
├── Makefile.in              # Legacy TEA build template
├── Makefile.local           # Verified local Linux build recipe
├── pkgIndex.tcl.in          # Legacy TEA package-index template
├── preparar_tea.tcl         # TEA preparation helper
├── API.md                   # Current API contract
├── BUILDING.md              # Build and test notes
└── LICENSE                  # MIT License
```

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style
2. Changes are well-tested
3. Documentation is updated
4. Commit messages are descriptive

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or feature requests:
- Check the troubleshooting section above
- Review llama.cpp documentation: https://github.com/ggerganov/llama.cpp
- Open an issue at https://github.com/rauleli/tclllama/issues

## Acknowledgments

- **llama.cpp**: Georgi Gerganov and contributors
- **Tcl**: The Tcl community
- **GGML**: Quantization and inference library
- **Ollama**: Model registry format and distribution platform used by the
  separate Tcl model manager

## Version History

The inherited documentation labeled an earlier API as `v1.0` and described
capabilities that are not an accurate inventory of the consolidated binding.
The code/package value `7.5` is a historical internal progression, while the
planned public release for the consolidated library is `1.0.0`. The old
singleton API examples have been replaced above. Notes on the superseded build
path are retained in [HISTORY.md](HISTORY.md); the previous README revision is
also available in Git history.

---

For the tested source revision, toolchain and model checksums, see
[BASELINE.md](BASELINE.md) and [CHARACTERIZATION.md](CHARACTERIZATION.md).

## Project support

Support the project through [Buy Me a Coffee](https://www.buymeacoffee.com/rauleli).

