# Tcl-Llama API Reference

Complete API reference for the Tcl-Llama extension.

## Module Loading

```tcl
package require tclllama
```

Or manually:

```tcl
load ./tclllama.so Tclllama
```

Once loaded, all `llama` commands become available.

## Command Reference

### Model Initialization and Cleanup

#### llama init

Initialize and load a model.

```tcl
llama init <model_path> ?options?
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_path | string | required | Path to GGUF format model file |
| options | dict | {} | Configuration options (see below) |

**Options Dictionary Keys:**

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| n_ctx | int | 128-32768 | 4096 | Context window size in tokens |
| n_batch | int | 32-2048 | 512 | Batch size for token processing |
| n_threads | int | 1-256 | system_count | Number of CPU threads |
| n_threads_batch | int | 1-256 | n_threads | Threads for batch processing |
| seed | int | -1, 0+ | -1 | Random seed (-1 for random) |
| rope_scaling_type | int | 0-2 | 0 | RoPE scaling method |
| rope_freq_base | float | 1.0-1000000 | 10000 | RoPE frequency base |
| rope_freq_scale | float | 0.001-1000 | 1.0 | RoPE frequency scale |
| yarn_ext_factor | float | 0.0-100 | 1.0 | YaRN extension factor |
| yarn_attn_factor | float | 0.0-10 | 1.0 | YaRN attention factor |
| yarn_beta_fast | float | 0.0-100 | 32.0 | YaRN beta fast |
| yarn_beta_slow | float | 1.0-10000 | 1.0 | YaRN beta slow |
| yarn_orig_ctx | int | 1-32768 | n_ctx | YaRN original context size |
| type_k | int | -1, 0-1 | -1 | KV cache type for keys |
| type_v | int | -1, 0-1 | -1 | KV cache type for values |
| gpu_layers | int | 0-999 | 0 | Layers to offload to GPU |
| main_gpu | int | 0-10 | 0 | Primary GPU index |
| tensor_split | string | "" | "" | GPU distribution for tensors |
| low_vram | bool | 0-1 | 0 | Enable low VRAM mode |
| mul_mat_q | bool | 0-1 | 1 | Use optimized matrix multiplication |
| logits_all | bool | 0-1 | 0 | Get logits for all tokens |
| embeddings | bool | 0-1 | 0 | Compute embeddings |
| offload_kqv | bool | 0-1 | 1 | Offload KQV to GPU |
| flash_attn | bool | 0-1 | 0 | Use flash attention |

**Returns:**
- TCL_OK on success
- TCL_ERROR on failure

**Example:**
```tcl
llama init "model.gguf" -n_ctx 2048 -n_threads 8
```

#### llama free

Unload the current model and free resources.

```tcl
llama free
```

**Parameters:** None

**Returns:**
- TCL_OK on success
- TCL_ERROR on failure (no model loaded)

**Example:**
```tcl
llama free
```

---

### Text Generation

#### llama generate

Generate text from a prompt.

```tcl
llama generate <prompt> ?options?
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| prompt | string | Input text to continue from |
| options | dict | Sampling and generation options |

**Generation Options:**

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| temperature | float | 0.0-2.0 | 0.8 | Token randomness |
| top_k | int | 1-1000 | 40 | Keep top-K most likely tokens |
| top_p | float | 0.0-1.0 | 0.95 | Cumulative probability threshold |
| min_p | float | 0.0-1.0 | 0.05 | Minimum probability for tokens |
| repeat_penalty | float | 0.0-2.0 | 1.1 | Penalty for token repetition |
| repeat_last_n | int | 0-2048 | 64 | Window for repeat penalty |
| frequency_penalty | float | -2.0-2.0 | 0.0 | Penalty based on token frequency |
| presence_penalty | float | -2.0-2.0 | 0.0 | Penalty for token presence |
| mirostat | int | 0-2 | 0 | Mirostat sampling (0=off, 1/2=on) |
| mirostat_tau | float | 0.0-10.0 | 5.0 | Mirostat target entropy |
| mirostat_eta | float | 0.0-1.0 | 0.1 | Mirostat learning rate |
| penalize_nl | bool | 0-1 | 1 | Penalize newline tokens |
| num_predict | int | -1, 1+ | -1 | Max tokens to generate (-1=infinite) |
| seed | int | -1, 0+ | -1 | Sampling random seed |

**Returns:**
- Generated text string

**Raises:**
- TCL_ERROR if no model is loaded
- TCL_ERROR if parameters are invalid

**Example:**
```tcl
# Simple generation
set response [llama generate "Hello, world" -num_predict 100]

# With sampling parameters
set response [llama generate "Explain AI" {
    temperature 0.7
    top_k 40
    top_p 0.9
    num_predict 256
}]

# Deterministic (temperature=0)
set response [llama generate "Q: What is 2+2?\nA:" {
    temperature 0.0
    num_predict 50
}]
```

**Temperature Effects:**
- `0.0` - Deterministic, always choose most likely token
- `0.5` - Low randomness, mostly predictable
- `1.0` - Balanced randomness
- `1.5` - High randomness, more creative
- `2.0` - Very high randomness, less coherent

**Token Limits:**
| Scenario | num_predict | Notes |
|----------|------------|-------|
| Short responses | 50-100 | Q&A, single sentences |
| Medium responses | 200-500 | Paragraphs, explanations |
| Long responses | 1000-2000 | Full articles |
| Unlimited | -1 | Use with caution |

---

### Tokenization

#### llama tokenize

Convert text to token IDs.

```tcl
llama tokenize <text> ?add_special?
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | string | required | Text to tokenize |
| add_special | bool | 1 | Include special tokens |

**Returns:**
- List of integer token IDs

**Example:**
```tcl
set tokens [llama tokenize "Hello, world!"]
puts $tokens  ;# Output: {1 Hello 29892 world 29991}

# Without special tokens
set tokens [llama tokenize "Hello" 0]
```

**Token ID Ranges:**
- 0-255 - Standard ASCII representation fallback
- 256+ - Model vocabulary tokens
- 65534 - Special token (often padding)
- 65535 - Unknown/out-of-vocabulary

#### llama detokenize

Convert token IDs back to text.

```tcl
llama detokenize <tokens> ?remove_special?
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tokens | list | required | List of integer token IDs |
| remove_special | bool | 0 | Remove special tokens from output |

**Returns:**
- Text string

**Example:**
```tcl
set text [llama detokenize {1 Hello 29892 world 29991}]
puts $text  ;# Output: " Hello, world!"

# Remove special tokens
set text [llama detokenize {1 Hello 29892} 1]
```

**Note:** Detokenized text may include whitespace padding from tokenization.

---

### Model Information

#### llama info

Get detailed information about the loaded model.

```tcl
llama info
```

**Parameters:** None

**Returns:**
Dictionary with keys:
| Key | Type | Description |
|-----|------|-------------|
| model_file | string | Path to loaded model file |
| context_size | int | Current context window size |
| vocabulary_size | int | Number of tokens in vocabulary |
| llama_version | string | llama.cpp version |
| architecture | string | Model architecture (e.g., "llama") |
| embedding_size | int | Size of embeddings |
| ffn_hidden_size | int | FFN hidden layer size |
| head_count | int | Number of attention heads |
| head_count_kv | int | Number of KV cache heads |
| layer_count | int | Number of transformer layers |
| rotary_dim | int | Rotary position embedding dimension |
| tokens_evaluated | int | Total tokens processed (input) |
| tokens_generated | int | Total tokens generated (output) |
| time_eval_ms | float | Time spent on token evaluation |
| time_gen_ms | float | Time spent on generation |

**Example:**
```tcl
set info [llama info]
puts "Model: [dict get $info model_file]"
puts "Context: [dict get $info context_size]"
puts "Vocab: [dict get $info vocabulary_size]"
puts "Layers: [dict get $info layer_count]"
puts "Tokens generated: [dict get $info tokens_generated]"
```

#### llama version

Get version and build information.

```tcl
llama version
```

**Parameters:** None

**Returns:**
Dictionary with keys:
| Key | Type | Description |
|-----|------|-------------|
| version | string | llama.cpp version string |
| build_commit | string | Git commit hash |
| build_number | int | Build number |

**Example:**
```tcl
set ver [llama version]
puts "llama.cpp [dict get $ver version]"
```

#### llama getcpuinfo

Get CPU information and detected features.

```tcl
llama getcpuinfo
```

**Parameters:** None

**Returns:**
Dictionary with keys:
| Key | Type | Description |
|-----|------|-------------|
| n_logical_cpus | int | Logical CPU count |
| n_physical_cpus | int | Physical CPU count |
| cpu_name | string | CPU model name |
| has_avx | bool | AVX support |
| has_avx2 | bool | AVX2 support |
| has_avx512 | bool | AVX-512 support |
| has_fma | bool | FMA support |
| has_sse3 | bool | SSE3 support |
| has_ssse3 | bool | SSSE3 support |
| has_sse4_1 | bool | SSE4.1 support |
| has_sse4_2 | bool | SSE4.2 support |

**Example:**
```tcl
set cpu [llama getcpuinfo]
puts "CPUs: [dict get $cpu n_logical_cpus]"
puts "AVX2: [dict get $cpu has_avx2]"
```

---

### Configuration

#### llama verbose

Set verbosity level for debug output.

```tcl
llama verbose <level>
```

**Parameters:**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| level | int | 0-3 | Verbosity level |

**Levels:**
- `0` - Silent (no output)
- `1` - Normal (errors only)
- `2` - Verbose (include info)
- `3` - Very verbose (include debug)

**Returns:**
- Previous verbosity level

**Example:**
```tcl
llama verbose 0  ;# Silent
llama generate "Test"

llama verbose 3  ;# Debug mode
llama generate "Test"
```

#### llama clearcache

Clear the KV (Key-Value) cache.

```tcl
llama clearcache
```

**Parameters:** None

**Returns:**
- TCL_OK

**Usage:**
Useful for:
- Freeing memory between inference runs
- Resetting context
- Improving performance with new prompts

**Example:**
```tcl
# After generation
llama generate "First prompt"

# Clear cache before new independent generation
llama clearcache

# New generation starts fresh
llama generate "Second prompt (no context from first)"
```

---

## Error Handling

All commands return TCL_OK on success or TCL_ERROR on failure.

**Common Errors:**

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "no model loaded" | No model initialized | Call `llama init` first |
| "model not found" | Model file missing | Check model path |
| "invalid parameter" | Bad option value | Check parameter ranges |
| "out of memory" | Insufficient RAM | Reduce context size |
| "failed to load" | Model corrupted | Verify model file |

**Example Error Handling:**
```tcl
if {[catch {llama init "model.gguf"} err]} {
    puts "Error loading model: $err"
    exit 1
}

if {[catch {llama generate "Test"} result]} {
    puts "Generation error: $result"
} else {
    puts "Result: $result"
}

llama free
```

---

## Performance Considerations

### Memory Usage

**Approximate Memory per Model:**
```
Memory ≈ Model Size + (Context Size × Hidden Size × 2)

For 7B model with 4K context:
≈ 7GB + (4K × 4096 × 2 bytes)
≈ 7GB + 32MB
≈ 7.03GB RAM
```

### Token Generation Speed

**Typical Performance (7B model, CPU):**
- First token: 1-5 seconds (prompt processing)
- Subsequent tokens: 0.5-2 seconds each (generation)
- Can vary significantly by CPU

**Optimization Tips:**
1. Reduce context size for faster initial processing
2. Use batch processing for multiple prompts
3. Reuse model across multiple generations
4. Adjust `top_k` to reduce candidate evaluation
5. Use `num_predict` to limit output

### Sampling Parameter Trade-offs

| Parameter | Speed Impact | Quality Impact |
|-----------|--------------|----------------|
| temperature | None | High |
| top_k | Medium | Medium |
| top_p | Low | Medium |
| repeat_penalty | Low | Low |
| mirostat | High | High |

---

## Examples

### Interactive Chat

```tcl
#!/usr/bin/tclsh

package require tclllama

# Load model
if {[catch {llama init "model.gguf" -n_ctx 2048} err]} {
    puts "Failed to load model: $err"
    exit 1
}

llama verbose 0
set history ""

while {1} {
    puts -nonewline "> "
    flush stdout

    if {[gets stdin line] < 0} break
    if {$line eq "exit"} break

    append history "User: $line\n"

    if {[catch {
        set response [llama generate $history -num_predict 256 -temperature 0.7]
    } err]} {
        puts "Error: $err"
        continue
    }

    puts "AI: $response\n"
    append history "AI: $response\n"
}

llama free
```

### Batch Processing

```tcl
#!/usr/bin/tclsh

package require tclllama
package require json

llama init "model.gguf" -n_ctx 1024 -n_threads 8

set input [open "prompts.jsonl" r]
set output [open "results.jsonl" w]

set count 0
while {[gets $input line] >= 0} {
    set obj [json::json2dict $line]
    set prompt [dict get $obj prompt]

    set result [llama generate $prompt -num_predict 200]

    dict set obj response $result
    puts $output [json::dict2json $obj]

    incr count
    if {$count % 10 == 0} {
        puts "Processed: $count"
        flush stdout
    }
}

close $input
close $output
llama free

puts "Complete: $count prompts processed"
```

### Model Comparison

```tcl
#!/usr/bin/tclsh

package require tclllama

set prompt "Explain quantum computing"
set models {"model1.gguf" "model2.gguf" "model3.gguf"}

set test_params {
    temperature 0.7
    top_k 40
    num_predict 200
}

foreach model $models {
    puts "\n=== $model ==="

    llama init $model -n_ctx 2048
    set info [llama info]
    puts "Params: [dict get $info layer_count] layers, [dict get $info vocabulary_size] vocab"

    set result [llama generate $prompt {*}$test_params]
    puts "Response: [string range $result 0 200]..."

    llama free
}
```

---

## Related Resources

- **README.md** - User guide and quick start
- **BUILDING.md** - Compilation instructions
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Model Hub**: https://huggingface.co/models?library=gguf

---

**Version**: 1.0
**Last Updated**: December 2024
**Compatibility**: Tcl 8.6+, llama.cpp latest
