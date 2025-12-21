# Changelog

All notable changes to Tcl-Llama are documented in this file.

## Format

This changelog follows [Keep a Changelog](https://keepachangelog.com/) principles.

Sections:
- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Now removed features
- **Fixed** - Bug fixes
- **Security** - Security vulnerability fixes
- **Documentation** - Documentation improvements

## [1.0] - 2024-12-21

### Added
- Initial release of Tcl-Llama
- Full llama.cpp integration for GGUF model inference
- Native Tcl extension (`tclllama.so`) for text generation
- Ik'nal v7.5 generation engine with advanced sampling
- Support for all GGUF-format models (Llama, Gemma, Mistral, Qwen, DeepSeek, etc.)
- Complete tokenization API:
  - `llama tokenize` - Convert text to token IDs
  - `llama detokenize` - Convert tokens back to text
- Advanced sampling parameters:
  - Temperature control (0.0-2.0)
  - Top-K sampling
  - Top-P (nucleus) sampling
  - Min-P threshold
  - Mirostat sampling (v1 and v2)
  - Repetition penalties (repeat_penalty, presence_penalty, frequency_penalty)
- Model management commands:
  - `llama init` - Load and initialize models
  - `llama free` - Unload models and free resources
  - `llama info` - Get detailed model information
  - `llama version` - Version information
  - `llama verbose` - Debug output control
- Utility commands:
  - `llama getcpuinfo` - CPU feature detection
  - `llama clearcache` - Clear KV cache
- SHA256 cryptographic utilities for model verification
- Ollama registry integration library (`library/ollama_registry.tcl`)
- TEA (Tcl Extension Architecture) build system
  - Autoconf configuration with automatic dependency detection
  - Support for custom llama.cpp paths
  - Automatic detection of:
    - llama.cpp headers (include/, ggml/include/)
    - libllama.so (in build/bin/, build/, lib/, or root)
    - OpenSSL libcrypto
- Comprehensive documentation:
  - README.md - Complete user guide
  - BUILDING.md - Detailed build instructions
  - API.md - Full API reference
  - CONTRIBUTING.md - Contribution guidelines
  - CHANGELOG.md - Version history
- MIT License for permissive use
- Performance metrics and telemetry
- Support for:
  - CPU inference
  - Optional GPU acceleration via llama.cpp
  - Cross-platform compilation (Linux, macOS, Windows)
  - Tcl 8.6+

### Build System Features
- Automatic dependency detection
- Support for multiple llama.cpp build directories:
  - User-specified path via `--with-llama`
  - Detection in standard locations (build/bin/, build/, lib/, root)
  - Automatic include path resolution (include/, ggml/include/)
- OpenSSL detection and linking
- Position-independent code compilation (-fPIC)
- Proper C++ compilation flags
- TCL package metadata generation

### Internals
- Complete C/C++ implementation with Tcl FFI
- UTF-8 validation for text processing
- Efficient batch processing for tokens
- Memory-safe resource management
- Error propagation to Tcl
- Proper reference counting for Tcl objects

---

## Future Roadmap (Planned)

### v1.1 (Planned)
- [ ] Streaming text generation API
- [ ] Batch processing optimizations
- [ ] GPU memory management improvements
- [ ] Additional sampling strategies
- [ ] Model quantization utilities
- [ ] Performance profiling tools

### v1.2 (Planned)
- [ ] Multi-model loading support
- [ ] Prompt caching for repeated queries
- [ ] Embedding generation API
- [ ] Fine-tuning support
- [ ] Custom sampler hooks

### v2.0 (Long-term)
- [ ] AsyncIO support for Tcl 8.7+
- [ ] Direct Python integration
- [ ] Web service mode
- [ ] Model serving infrastructure
- [ ] Distributed inference support

---

## Known Issues

### Current (v1.0)

#### Limitations
1. **Single Model Loading**: Only one model can be loaded at a time
   - Workaround: Use `llama free` before loading a new model

2. **Streaming Output**: Generation returns complete output
   - Workaround: Use `num_predict` to limit output size

3. **Context Limitation**: Cannot exceed model's maximum context
   - Workaround: Summarize long context before processing

#### Platform-Specific
- **Windows**: Requires proper OpenSSL configuration
- **macOS**: May need to set `DYLD_LIBRARY_PATH` for OpenSSL
- **ARM64**: Limited GPU support, CPU-only recommended

---

## Upgrade Guide

### From v0.x to v1.0

Complete rewrite with new API. Existing code needs migration:

**Old Code:**
```tcl
llama_init model.gguf
set result [llama_generate "prompt"]
llama_cleanup
```

**New Code:**
```tcl
llama init model.gguf
set result [llama generate "prompt"]
llama free
```

See README.md for full API documentation.

---

## Contributors

### v1.0 Release
- Core development and implementation
- Documentation and examples
- Testing and bug fixes
- Build system setup

---

## Acknowledgments

### Dependencies
- **llama.cpp** - Efficient GGUF inference (Georgi Gerganov)
- **Tcl/Tk** - The Tcl scripting language and community
- **GGML** - Machine learning library for inference
- **OpenSSL** - Cryptographic operations

### Inspiration
- Ollama project for model distribution patterns
- llama.cpp for production-ready inference
- Tcl extension best practices

---

## Version History Summary

| Version | Date | Status | Type |
|---------|------|--------|------|
| 1.0 | 2024-12-21 | Release | Initial |

---

## How to Report Issues

Found a bug or want to suggest a feature?

1. Check [existing issues](https://github.com/your-org/tcl-llama/issues)
2. Open a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, Tcl version, llama.cpp version)
   - Error messages or logs

---

## License

All changes are licensed under the MIT License. See LICENSE file for details.

---

## Release Schedule

- **Patch releases (v1.0.x)**: As needed for critical fixes
- **Minor releases (v1.x.0)**: Every 2-3 months with features
- **Major releases (vX.0.0)**: As warranted for major changes

---

**Last Updated**: 2024-12-21
**Next Planned Release**: v1.1 (ETA: Q1 2025)
