# Contributing to Tcl-Llama

Thank you for your interest in contributing to Tcl-Llama! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome diverse perspectives and experiences
- Focus on constructive feedback
- Report unacceptable behavior to project maintainers

## How to Contribute

### 1. Report Bugs

**Before submitting a bug report:**
- Check if the issue already exists
- Verify you're using the latest version
- Try to reproduce the issue

**When submitting a bug report, include:**

```markdown
**Description:**
Clear description of the bug

**Steps to Reproduce:**
1. Load model X
2. Run command Y
3. Observe error Z

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: (e.g., Linux Fedora 41, macOS 13.0)
- Tcl Version: (e.g., 8.6.11)
- llama.cpp Version: (commit hash or date)
- gcc Version: (e.g., 14.3.1)

**Error Message:**
Full error output or traceback

**Model Used:**
Which model file and size
```

### 2. Suggest Enhancements

**Enhancement request template:**

```markdown
**Description:**
Clear description of proposed enhancement

**Motivation:**
Why would this be useful?

**Proposed Solution:**
How should it work?

**Alternatives Considered:**
Other approaches you've thought of

**Example Usage:**
How would users interact with this?
```

### 3. Contribute Code

#### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/tcl-llama.git
cd tcl-llama

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/tcl-llama.git
```

#### Create Feature Branch

```bash
# Update main branch
git fetch upstream
git rebase upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

**Branch naming conventions:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `perf/description` - Performance improvements

#### Make Your Changes

**Code Style Guidelines:**

1. **C/C++ Code** (`generic/tclllama.c`, `generic/sha256.c`):
   - Use 4-space indentation (not tabs)
   - Follow Linux kernel style where applicable
   - Keep lines under 100 characters
   - Use meaningful variable names
   - Add comments for complex logic
   - Include error handling

   **Example:**
   ```c
   static int Llama_Generate_Cmd(ClientData cd, Tcl_Interp *interp,
                                  int objc, Tcl_Obj *const objv[]) {
       /* Validate argument count */
       if (objc < 2) {
           Tcl_WrongNumArgs(interp, 1, objv, "prompt ?options?");
           return TCL_ERROR;
       }

       /* Get prompt text */
       const char *prompt = Tcl_GetString(objv[1]);
       if (!prompt) {
           return TCL_ERROR;
       }

       /* ... implementation ... */
       return TCL_OK;
   }
   ```

2. **Tcl Code** (`library/*.tcl`):
   - Use 4-space indentation
   - Follow Tcl style conventions
   - Use meaningful procedure names
   - Document procedures with comments

   **Example:**
   ```tcl
   # Query available models from Ollama registry
   #
   # Arguments:
   #   url      - Ollama registry URL
   #
   # Returns:
   #   List of model dictionaries
   proc registry::list_models {url} {
       set models {}
       # ... implementation ...
       return $models
   }
   ```

3. **Documentation** (`*.md`):
   - Use clear, concise language
   - Include code examples
   - Keep formatting consistent
   - Update table of contents for major changes

4. **Commit Messages**:
   ```
   Short, imperative summary (50 chars max)

   More detailed explanation (72 chars per line max) if needed.
   - Use bullet points for lists
   - Reference issues with #123
   - Explain why, not what

   Fixes #123
   ```

   **Good examples:**
   ```
   Fix segmentation fault in tokenizer
   Improve error handling for invalid models
   Add GPU acceleration support
   Update documentation for Windows builds
   ```

   **Bad examples:**
   ```
   Fixed bug
   Changes
   WIP
   asdf
   ```

#### Test Your Changes

**Build locally:**

```bash
# Clean previous build
make clean

# Rebuild
./configure --with-llama=/path/to/llama.cpp
make

# Test the build
tclsh << 'EOF'
load ./tclllama.so Tclllama
puts "Extension loaded successfully"
EOF
```

**Test your specific changes:**

```bash
# Create a test script for your feature
# test_feature.tcl

load ./tclllama.so Tclllama
llama init "model.gguf"

# Test your new/modified code here
set result [llama generate "Test prompt"]
puts "Result: $result"

llama free
puts "Test passed!"
```

**Run any existing tests:**

```bash
# If tests directory exists
cd tests
tclsh run_tests.tcl
```

#### Push and Create Pull Request

```bash
# Commit your changes
git add .
git commit -m "Your descriptive commit message"

# Push to your fork
git push origin feature/your-feature-name
```

**On GitHub:**
1. Go to your fork
2. Click "Compare & pull request"
3. Fill in the PR template with:
   - Description of changes
   - Related issues (use #123 format)
   - Type of change (fix, feature, etc.)
   - How to test the changes
   - Checklist completion

### 4. Documentation Contributions

**Areas needing documentation:**
- API documentation
- Usage examples
- Troubleshooting guides
- Architecture documentation
- Build instructions for specific platforms

**To contribute documentation:**

```bash
# Edit markdown files
vim README.md  # Main documentation
vim BUILDING.md  # Build instructions
vim API.md  # API reference (if exists)

# Verify formatting
# Commit and push as above
```

## Development Setup

### Local Development Environment

```bash
# Create development directory
mkdir -p ~/tcl-llama-dev
cd ~/tcl-llama-dev

# Clone your fork
git clone https://github.com/YOUR_USERNAME/tcl-llama.git
cd tcl-llama

# Set up development build
autoconf
./configure --with-llama=/path/to/llama.cpp
make

# Create symlink for easy testing
ln -s $(pwd)/tclllama.so ~/.tcllama_dev

# In Tcl scripts, use:
# load ~/.tcllama_dev/tclllama.so Tclllama
```

### Debugging

**Compile with debug symbols:**

```bash
CFLAGS="-g -O0" CXXFLAGS="-g -O0" ./configure --with-llama=/path/to/llama.cpp
make clean && make
```

**Debug with gdb:**

```bash
gdb --args tclsh test.tcl

# Inside gdb:
(gdb) break Llama_Generate_Cmd
(gdb) run
(gdb) next
(gdb) print variable_name
(gdb) bt  # Backtrace on crash
```

**Verbose output:**

```tcl
llama verbose 3  # Maximum verbosity
# Then run your test code
```

## Project Structure for Contributors

```
tcl-llama/
├── generic/                    # C/C++ source code
│   ├── tclllama.c             # Main extension (edit here)
│   └── sha256.c               # Crypto utilities
├── library/                    # Tcl libraries
│   └── ollama_registry.tcl    # Ollama integration
├── tests/                      # Test scripts
├── doc/                        # Documentation
├── configure.ac               # Build configuration (autoconf)
├── Makefile.in               # Build template
├── README.md                 # Main documentation
├── BUILDING.md               # Build instructions
├── CONTRIBUTING.md           # This file
└── LICENSE                   # MIT License
```

**Key files to understand:**
- `configure.ac` - Autoconf configuration (build system)
- `Makefile.in` - Makefile template
- `generic/tclllama.c` - Core extension logic
- `README.md` - User documentation

## Review Process

1. **Automated Checks**: Tests run automatically
2. **Code Review**: Maintainers review for:
   - Code quality and style
   - Performance implications
   - Documentation updates
   - Test coverage
3. **Feedback**: Address review comments
4. **Approval**: Once approved, PR is merged

## Common Contributions

### Adding a New Command

1. Add command handler in `generic/tclllama.c`:

```c
static int Llama_MyCommand_Cmd(ClientData cd, Tcl_Interp *interp,
                                int objc, Tcl_Obj *const objv[]) {
    /* Implementation */
    return TCL_OK;
}
```

2. Register in `Tclllama_Init()`:

```c
Tcl_CreateObjCommand(interp, "llama_mycommand",
                     Llama_MyCommand_Cmd, NULL, NULL);
```

3. Document in README.md API section
4. Add test case

### Improving Error Handling

1. Identify error scenario
2. Add validation check
3. Set descriptive error message
4. Return TCL_ERROR
5. Test error path
6. Document in error handling section

### Performance Optimization

1. Profile existing code
2. Identify bottleneck
3. Implement optimization
4. Benchmark improvement
5. Document changes
6. Ensure no functionality loss

## Useful Commands

```bash
# View git log
git log --oneline -10

# See changes before committing
git diff

# Unstage changes
git reset HEAD <file>

# Undo last commit (uncommitted changes)
git reset --soft HEAD~1

# View branches
git branch -a

# Clean up merged branches
git branch -d <branch-name>

# Fetch latest updates
git fetch upstream
git rebase upstream/main
```

## Questions?

- Check existing issues and discussions
- Read documentation thoroughly
- Ask maintainers in issues or discussions
- Review similar contributions

## Recognition

Contributors will be recognized:
- In commit history
- In project README acknowledgments
- In release notes for significant contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Tcl-Llama! Your efforts help make this project better for everyone.

**Version**: 1.0
**Last Updated**: December 2024
