# Security Policy

## Reporting Security Vulnerabilities

**Do NOT open public GitHub issues for security vulnerabilities.**

If you discover a security vulnerability in Tcl-Llama, please report it responsibly:

1. **Email**: Send details to the project maintainers (check repository)
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
3. **Timeline**: Allow 90 days for a fix before public disclosure
4. **Confidentiality**: Do not discuss publicly until a fix is released

## Security Best Practices for Users

### Handling API Keys and Credentials

**Never commit or share:**
- API keys
- Authentication tokens
- Model files with sensitive training data
- System credentials
- Personal information

**When generating from code:**

```tcl
# ❌ DON'T - Expose credentials
llama generate "Use API key: sk-1234567890abcdef"

# ✅ DO - Use environment variables
set api_key $::env(API_KEY)
```

### Model Files

**Security considerations:**

1. **Verify Model Integrity**
   ```bash
   # Check SHA256 before loading
   sha256sum model.gguf
   ```

2. **Source Only Trusted Models**
   - Download from official sources only
   - Hugging Face, Ollama, or known providers
   - Verify checksums

3. **Storage**
   - Keep models on secure, encrypted storage
   - Restrict file permissions: `chmod 600 model.gguf`
   - Don't store in shared directories

### Input Validation

**Sanitize user inputs before processing:**

```tcl
# ✅ DO - Validate input
proc safe_generate {prompt} {
    # Validate prompt length
    if {[string length $prompt] > 10000} {
        error "Prompt too long"
    }

    # Validate character encoding
    if {![string is utf8 -strict $prompt]} {
        error "Invalid UTF-8 encoding"
    }

    return [llama generate $prompt]
}
```

### Information Disclosure Prevention

**What to keep private:**

1. **System Paths**
   - Don't include `/home/username` in examples
   - Use placeholders like `/path/to/`
   - Use environment variables instead

2. **Build Information**
   - Don't commit `config.log` or `config.status`
   - Don't commit generated `Makefile`
   - Use `.gitignore` (included in repository)

3. **Compiler Information**
   - Don't expose full GCC/Clang versions in production
   - Use `-DNDEBUG` for release builds

4. **Model Information**
   - Don't log complete model paths
   - Don't expose model parameter details to untrusted sources

## Security in Dependencies

### llama.cpp

- Keep llama.cpp updated to latest version
- Review llama.cpp security advisories
- Build with optimization flags (`-O3`)

### OpenSSL

- Ensure libcrypto 1.1.0 or later (1.0.2 is deprecated)
- Keep OpenSSL updated
- Verify: `openssl version`

### Tcl

- Use Tcl 8.6.11 or later (security updates)
- Verify: `tclsh --version`

## Secure Coding

### Memory Safety

**C Code Considerations:**

```c
/* ❌ DON'T - Buffer overflow risk */
char buffer[256];
strcpy(buffer, user_input);  // Dangerous!

/* ✅ DO - Safe bounds checking */
char buffer[256];
strncpy(buffer, user_input, sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';
```

### Error Messages

**Don't expose sensitive information in errors:**

```tcl
# ❌ DON'T - Exposes system paths
if {[catch {llama init $model_path} err]} {
    puts "Error: $err"  # Might reveal /home/username
}

# ✅ DO - Generic error message
if {[catch {llama init $model_path} err]} {
    puts "Error: Failed to load model"
    # Log detailed error only to file
}
```

## Build Security

### Compilation Flags

**Recommended flags for security:**

```bash
# Security flags
CFLAGS="-O3 -D_FORTIFY_SOURCE=2 -fstack-protector-strong"
CXXFLAGS="-O3 -D_FORTIFY_SOURCE=2 -fstack-protector-strong"
LDFLAGS="-Wl,-z,relro,-z,now"

# Position independent code
CFLAGS="-fPIC"
```

### Strip Debugging Symbols

**For release builds, remove debug symbols:**

```bash
strip tclllama.so
```

This reduces binary size and removes symbol information.

### Don't Commit Build Files

**.gitignore covers:**
- `config.log` - Contains build system paths
- `config.status` - Build configuration
- `Makefile` - Generated with absolute paths
- `*.o` - Object files
- `tclllama.so` - Compiled binary
- `autom4te.cache/` - Autoconf cache

## Runtime Security

### Limit Resource Usage

**Prevent denial of service:**

```tcl
# Limit context size
llama init "model.gguf" -n_ctx 2048  # Don't use huge contexts

# Limit token generation
llama generate "prompt" -num_predict 1000  # Set reasonable limits

# Use timeouts (not yet implemented, but plan for it)
```

### Disable Debug Output in Production

```tcl
# Development
llama verbose 3

# Production
llama verbose 0  # Silent mode
```

## Documentation Security

### What NOT to Include

❌ **Never include in commits:**
- Full system paths `/home/username/...`
- API keys or tokens
- Email addresses
- Personal information
- SSH keys or credentials
- Database URLs with passwords

✅ **Use instead:**
- Placeholders: `/path/to/model`
- Environment variables: `$HOME`, `$USER`
- Generic examples: `/home/user`
- Paths without usernames

### Example: Safe vs Unsafe

**Unsafe (Exposes personal info):**
```
./configure --with-llama=/home/rauleli/llama.cpp
```

**Safe (Generic path):**
```
./configure --with-llama=/path/to/llama.cpp
```

## Third-Party Dependencies

### Review Before Using

1. Check dependency security advisories
2. Use pinned versions in requirements
3. Monitor dependencies for vulnerabilities
4. Use `ldd tclllama.so` to verify dependencies

### Current Dependencies

| Library | Version | Purpose | Security Note |
|---------|---------|---------|---------------|
| llama.cpp | Latest | Inference | Keep updated |
| OpenSSL | 1.1+ | Cryptography | Use 1.1.1+ |
| Tcl | 8.6+ | Scripting | Keep updated |

## Vulnerability Types to Avoid

### In C/C++ Code

- **Buffer overflows** - Use safe string functions
- **Format strings** - Use proper format specifiers
- **Integer overflows** - Check bounds before operations
- **Use after free** - Properly manage memory
- **SQL injection** - N/A (no SQL in this project)

### In Tcl Code

- **Command injection** - Use `list` instead of string concatenation
- **Path traversal** - Validate paths
- **Information disclosure** - Don't expose system paths
- **Resource exhaustion** - Limit loops and recursion

## Testing for Security

### Security Checklist

Before release:

- [ ] No hardcoded credentials in code
- [ ] No system paths in documentation
- [ ] `.gitignore` prevents build artifacts
- [ ] No debug symbols in release binary
- [ ] Compiler warnings clean (`-Wall -Wextra`)
- [ ] Dependencies updated
- [ ] Error messages generic
- [ ] Input validation in place
- [ ] Resource limits enforced
- [ ] No sensitive data in logs

### Manual Review

```bash
# Check for credentials
grep -r "password\|secret\|token\|key" src/ --include="*.c" --include="*.tcl"

# Check for hardcoded paths
grep -r "^/home\|^/Users\|^/root" . --include="*.c" --include="*.tcl"

# Check for debug calls in production code
grep -r "printf.*debug\|fprintf.*stderr" src/ --include="*.c"
```

## Security Through Obscurity Warning

This project relies on:
1. **Code quality** - Well-written, maintainable code
2. **Open source review** - Community can audit
3. **Dependency security** - Keeping dependencies updated
4. **Input validation** - Proper bounds checking
5. **Good practices** - Following security guidelines

**NOT on:**
- Hiding security issues
- Trusting users blindly
- Assuming security through obscurity

## Responsible Disclosure Examples

### Good Report

```
Subject: Information Disclosure in config.log

Description:
config.log contains full system paths including usernames.
This is generated during build and should not be committed.

Impact:
An attacker who obtains config.log can determine:
- System architecture
- User directory structure
- Build configuration
- Software versions

Reproduction:
1. Run ./configure --with-llama=/path/to/llama
2. Examine config.log
3. Note presence of /home/username paths

Suggested Fix:
Add config.log to .gitignore
```

### Bad Report

```
Subject: URGENT!!!1 SECURITY HOLE!!!

your project is insecure
```

## Timeline for Security Fixes

1. **Day 0**: Report received
2. **Days 1-7**: Investigation and confirmation
3. **Days 8-30**: Fix development and testing
4. **Days 31-60**: Backport to previous versions (if needed)
5. **Day 61**: Security release published
6. **Day 90**: Public disclosure (if not already fixed)

## Changelog

### Current Version (1.0)

- ✅ Added `.gitignore` to prevent credential leaks
- ✅ Added SECURITY.md documentation
- ✅ Input validation in core functions
- ✅ Safe string handling in C code
- ✅ Generic paths in documentation

### Future Enhancements

- [ ] Security scanning in CI/CD pipeline
- [ ] SBOM (Software Bill of Materials) generation
- [ ] Formal security audit
- [ ] Hardened build options
- [ ] Security policy automation

## Contact

For security inquiries, contact the project maintainers directly (not through public issues).

---

**Last Updated**: December 21, 2024
**Status**: Active
**Policy Version**: 1.0

This security policy is maintained to protect users and the integrity of Tcl-Llama.
