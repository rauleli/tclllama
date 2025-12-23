# Tcl-Llama: LLM Integration for Tcl

A Tcl extension that integrates **llama.cpp** for running large language models (LLMs) directly in Tcl scripts. Supports all GGUF-format models including Llama, Gemma, Mistral, Qwen, DeepSeek, and more.

## Features

- **Native Tcl Integration**: Use LLMs directly from Tcl scripts without external processes
- **Universal Model Support**: Compatible with any GGUF-format model via llama.cpp
- **Advanced Sampling**: Temperature, Top-K, Top-P, Min-P, Mirostat, and more
- **Tokenization**: Direct access to tokenize/detokenize operations
- **Performance Metrics**: Built-in telemetry for inference timing and token counts
- **SHA256 Support**: Cryptographic hash operations for model verification
- **Cross-Platform**: Works on Linux, macOS, and Windows (with appropriate compilation)

## Prerequisites

- **Tcl 8.6+** (with development headers)
- **llama.cpp** compiled as a shared library (`libllama.so`)
- **OpenSSL** (libcrypto) for cryptographic operations
- **GCC/Clang** C++ compiler
- **Autoconf/Automake** (for building from source)

### Installation of Prerequisites (Fedora/RHEL)

```bash
dnf install tcl-devel openssl-devel gcc-c++ autoconf automake libtool
```

### Installation of Prerequisites (Debian/Ubuntu)

```bash
apt-get install tcl-dev libssl-dev g++ autoconf automake libtool
```

### Building llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
cmake --build . --config Release
```

This will create `libllama.so` in `build/bin/` directory.

## Building Tcl-Llama

### Step 1: Generate Configuration

```bash
autoconf
```

### Step 2: Run Configure

Point to your llama.cpp build directory:

```bash
./configure --with-llama=/path/to/llama.cpp
```

The script will automatically detect:
- llama.cpp headers (in `include/` and `ggml/include/`)
- `libllama.so` (in `build/bin/`, `build/`, `lib/`, or root)
- OpenSSL libcrypto

### Step 3: Build and Install

```bash
make
make install  # optional; places library in /usr/local/lib
```

This creates:
- `tclllama.so` - The compiled Tcl extension
- `pkgIndex.tcl` - Tcl package metadata

## Quick Start

### Loading the Extension

```tcl
package require tclllama

# Or manually load from current directory:
load ./tclllama.so Tclllama
```

### Basic Example: Generate Text

```tcl
# Initialize a model
set model_path "/path/to/model.gguf"
llama init $model_path -n_ctx 2048

# Generate text
set response [llama generate "Hello, how are you?" -num_predict 100]
puts $response

# Cleanup
llama free
```

### Advanced Example: Custom Sampling Parameters

```tcl
llama init "model.gguf" -n_ctx 4096

set options {
    temperature 0.7
    top_k 40
    top_p 0.95
    num_predict 256
}

set response [llama generate "Explain quantum computing in simple terms" {*}$options]
puts $response

llama free
```

## API Reference

### Initialization and Cleanup

#### `llama init <model_path> [options]`

Load a GGUF model and initialize the inference engine.

**Arguments:**
- `model_path` - Path to GGUF model file (required)
- `options` - Dictionary of configuration options (optional)

**Options:**
- `n_ctx` - Context window size (default: 4096)
- `n_batch` - Batch size for processing (default: 512)
- `n_threads` - Number of CPU threads (default: system count)
- `seed` - Random seed (default: -1 for random)
- `gpu_layers` - Number of layers to offload to GPU (default: 0)

**Example:**
```tcl
llama init "model.gguf" -n_ctx 2048 -n_threads 8
```

#### `llama free`

Unload the model and free all resources.

**Example:**
```tcl
llama free
```

### Text Generation

#### `llama generate <prompt> [options]`

Generate text based on a prompt.

**Arguments:**
- `prompt` - Input text (required)
- `options` - Sampling parameters (optional)

**Sampling Options:**
- `temperature` - Randomness (0.0-2.0, default: 0.8)
  - Lower = more deterministic
  - Higher = more creative
- `top_k` - Top-K sampling (default: 40)
  - Limits to top K most likely tokens
- `top_p` - Top-P/Nucleus sampling (0.0-1.0, default: 0.95)
  - Cumulative probability threshold
- `min_p` - Minimum probability (0.0-1.0, default: 0.05)
- `repeat_penalty` - Repetition penalty (default: 1.1)
  - > 1.0 discourages token repetition
- `repeat_last_n` - Tokens to apply penalty to (default: 64)
- `presence_penalty` - Presence penalty (default: 0.0)
- `frequency_penalty` - Frequency penalty (default: 0.0)
- `mirostat` - Mirostat sampling (0=disabled, 1 or 2, default: 0)
- `mirostat_tau` - Mirostat target entropy (default: 5.0)
- `mirostat_eta` - Mirostat learning rate (default: 0.1)
- `num_predict` - Max tokens to generate (-1 = infinite, default: -1)
- `seed` - Sampling seed (default: -1)

**Example:**
```tcl
set result [llama generate "What is AI?" {
    temperature 0.7
    top_k 40
    top_p 0.9
    num_predict 200
}]
puts $result
```

### Tokenization

#### `llama tokenize <text>`

Convert text to token IDs.

**Arguments:**
- `text` - Input text

**Returns:**
List of integer token IDs

**Example:**
```tcl
set tokens [llama tokenize "Hello world"]
puts $tokens
# Output: 1 Hello 2919 world
```

#### `llama detokenize <token_list>`

Convert token IDs back to text.

**Arguments:**
- `token_list` - List of integer token IDs

**Returns:**
Text string

**Example:**
```tcl
set text [llama detokenize {1 Hello 2919 world}]
puts $text
# Output: Hello world
```

### Model Information

#### `llama version`

Get version information.

**Returns:**
Dictionary with version and build details

**Example:**
```tcl
set version [llama version]
puts $version
```

#### `llama info`

Get detailed information about the loaded model.

**Returns:**
Dictionary with model statistics:
- `model_file` - Path to loaded model
- `context_size` - Context window size
- `vocabulary_size` - Number of tokens in vocabulary
- `llama_version` - llama.cpp version
- `tokens_generated` - Total tokens generated
- `tokens_processed` - Total tokens processed

**Example:**
```tcl
set info [llama info]
dict get $info vocabulary_size
```

#### `llama getcpuinfo`

Get CPU information and capabilities.

**Returns:**
Dictionary with CPU details (threads, features, etc.)

### Configuration

#### `llama verbose <level>`

Set verbosity level for debugging.

**Arguments:**
- `level` - 0=silent, 1=normal, 2=verbose, 3=very verbose

**Example:**
```tcl
llama verbose 2
```

#### `llama clearcache`

Clear the KV cache to free memory.

**Example:**
```tcl
llama clearcache
```

## Tcl Library: ollama_registry.tcl

The suite includes `library/ollama_registry.tcl` - a Tcl library for managing Ollama model registries.

### Features

- Registry querying and model listing
- Model pull/push operations
- Connection timeout handling (30s per chunk)

### Usage

```tcl
source library/ollama_registry.tcl

# Query Ollama registry
set models [registry::list_models "http://localhost:11434"]
foreach model $models {
    puts "Model: [dict get $model name]"
}
```

## Performance Tips

1. **Batch Processing**: Group prompts together for better throughput
   ```tcl
   foreach prompt $prompt_list {
       set result [llama generate $prompt -num_predict 100]
       # Process result
   }
   ```

2. **Reuse Models**: Load once and reuse rather than init/free repeatedly
   ```tcl
   llama init "model.gguf" -n_ctx 4096
   # Many generations...
   llama free
   ```

3. **Control Context**: Use appropriate context size
   - Larger = more memory, better quality
   - Smaller = faster, less memory

4. **Temperature Tuning**:
   - 0.0 = deterministic (good for Q&A)
   - 0.7-0.9 = balanced
   - 1.5+ = very creative

5. **Token Limits**: Set `num_predict` to reasonable limits
   ```tcl
   llama generate "Question" -num_predict 256
   ```

## Examples

### Interactive Chat

```tcl
#!/usr/bin/tclsh

package require tclllama

# Load model
llama init "model.gguf" -n_ctx 2048

set history ""

while {1} {
    puts -nonewline "> "
    flush stdout
    gets stdin user_input

    if {$user_input eq "exit"} break

    append history "User: $user_input\n"

    set response [llama generate $history -num_predict 200]
    puts "AI: $response\n"

    append history "AI: $response\n"
}

llama free
```

### Batch Processing

```tcl
#!/usr/bin/tclsh

package require tclllama

set input_file [open "prompts.txt" r]
set output_file [open "results.txt" w]

llama init "model.gguf" -n_ctx 1024

while {[gets $input_file line] >= 0} {
    if {[string length $line] > 0} {
        set result [llama generate $line -num_predict 150]
        puts $output_file "Input: $line"
        puts $output_file "Output: $result"
        puts $output_file "---"
    }
}

close $input_file
close $output_file
llama free
```

### Model Information Query

```tcl
package require tclllama

llama init "model.gguf"

set info [llama info]
puts "Model File: [dict get $info model_file]"
puts "Context Size: [dict get $info context_size]"
puts "Vocabulary Size: [dict get $info vocabulary_size]"

set version [llama version]
puts "llama.cpp Version: [dict get $version version]"

llama free
```

## Troubleshooting

### Error: "libllama.so: cannot open shared object file"

**Solution:** Ensure llama.cpp is built with `-DBUILD_SHARED_LIBS=ON` and add to library path:
```bash
export LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH
```

### Error: "ggml.h: No such file or directory"

**Solution:** Verify configure was run with correct `--with-llama` path pointing to llama.cpp root directory.

### Error: "Segmentation fault" during generation

**Solution:**
- Check context size is reasonable
- Verify model file is not corrupted
- Run with `llama verbose 2` for debugging

### Slow Performance

**Solution:**
- Reduce context size
- Lower `top_k` value (fewer candidates to evaluate)
- Set `num_predict` to reasonable limit
- Use CPU-specific optimizations in llama.cpp build

## Architecture

```
┌─────────────────────────────────────┐
│   Tcl Scripts (.tcl)                │
├─────────────────────────────────────┤
│   Tcl-Llama Extension (tclllama.so) │
│   ├─ Generation (Ik'nal v7.5)      │
│   ├─ Tokenization                  │
│   └─ Model Management              │
├─────────────────────────────────────┤
│   llama.cpp (libllama.so)           │
│   ├─ GGML Operations               │
│   ├─ Sampling & Inference          │
│   └─ Quantization Support          │
├─────────────────────────────────────┤
│   Hardware                          │
│   ├─ CPU (x86_64, ARM)             │
│   └─ Optional GPU (via llama.cpp)   │
└─────────────────────────────────────┘
```

## Project Structure

```
tcl-llama/
├── generic/
│   ├── tclllama.c           # Main extension implementation
│   └── sha256.c             # Cryptographic utilities
├── library/
│   └── ollama_registry.tcl  # Ollama integration library
├── tclconfig/               # Tcl autoconf macros
├── configure.ac             # Autoconf configuration
├── Makefile.in              # Build template
├── pkgIndex.tcl.in          # Package metadata template
├── README.md                # This file
├── LICENSE                  # MIT License
└── doc/
    └── ...                  # Additional documentation
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
- Open an issue on the project repository

## Acknowledgments

- **llama.cpp**: Georgi Gerganov and contributors
- **Tcl**: The Tcl community
- **GGML**: Quantization and inference library
- **Ollama**: Model distribution platform (integrated library)

## Version History

- **v1.0** (Current)
  - Initial release with full llama.cpp integration
  - Ik'nal v7.5 generation engine
  - Universal model support (GGUF format)
  - Advanced sampling options
  - SHA256 cryptography support

---

## ☕ Support my work

If this project has been helpful to you or saved you some development time, consider buying me a coffee! Your support helps me keep exploring new optimizations and sharing quality code.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/rauleli)

---

**Last Updated:** December 2025
**Compatibility:** Tcl 8.6+, llama.cpp latest, OpenSSL 1.1+

