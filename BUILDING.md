# Building Tcl-Llama from Source

This document provides comprehensive instructions for building Tcl-Llama from source code.

## System Requirements

### Minimum Requirements
- **Tcl**: 8.6 or later (with development headers)
- **C++ Compiler**: GCC 7.0+ or Clang 5.0+
- **CMake**: 3.12+ (for building llama.cpp)
- **OpenSSL**: 1.1.0+ (libcrypto)
- **llama.cpp**: Latest version

### Recommended Setup
- **OS**: Linux (x86_64 or ARM64), macOS, Windows (with MinGW/MSVC)
- **RAM**: 8GB+ (for running models)
- **Disk**: 20GB+ (for models and build artifacts)

## Step-by-Step Build Instructions

### 1. Install Build Tools and Dependencies

#### Fedora/RHEL/CentOS

```bash
dnf install -y \
    tcl-devel \
    openssl-devel \
    gcc-c++ \
    cmake \
    make \
    autoconf \
    automake \
    libtool \
    git
```

#### Debian/Ubuntu

```bash
apt-get update
apt-get install -y \
    tcl-dev \
    libssl-dev \
    build-essential \
    cmake \
    make \
    autoconf \
    automake \
    libtool \
    git
```

#### macOS (with Homebrew)

```bash
brew install tcl openssl cmake autoconf automake libtool
```

#### Windows (MSVC)

Download and install:
- Visual Studio Build Tools or Visual Studio Community
- CMake (https://cmake.org/download/)
- ActiveTcl or Tcl/Tk distribution
- OpenSSL development package

### 2. Verify Tcl Installation

```bash
tclsh --version
# Expected output: tclsh8.6 (or higher)

# Verify Tcl headers are available
ls /usr/include/tcl.h  # Linux/macOS
# or
find "C:\Program Files" -name tcl.h  # Windows
```

### 3. Build llama.cpp

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Create build directory
mkdir -p build
cd build

# Configure with shared library support
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_FLAGS="-O3"

# Compile (use appropriate number of cores)
cmake --build . --config Release -j$(nproc)

# Verify libllama.so was created
ls -lh bin/libllama.so*
# Expected output: libllama.so, possibly libllama.so.X.Y

cd ../..  # Go back to parent directory
```

**Optional: GPU Support**

If you want GPU acceleration, add to cmake command:

```bash
# For NVIDIA CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native

# For AMD HIP
cmake .. -CMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_HIP=ON

# For Metal (macOS)
cmake .. -CMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_METAL=ON
```

### 4. Clone and Prepare Tcl-Llama

```bash
# Clone tcl-llama repository
git clone <tcl-llama-repository-url> tcl-llama
cd tcl-llama

# Verify directory structure
ls -la
# Expected: generic/tclllama.c, library/, configure.ac, Makefile.in, etc.
```

### 5. Generate Configure Script

```bash
# From tcl-llama root directory
autoconf

# Verify configure script was created
ls -lh configure
# Expected: ~30KB executable file
```

**Troubleshooting autoconf:**

If autoconf fails, ensure you have m4:
```bash
# Fedora/RHEL
dnf install m4

# Debian/Ubuntu
apt-get install m4

# macOS
brew install m4
```

### 6. Run Configure Script

```bash
# Basic configuration
./configure --with-llama=/path/to/llama.cpp

# Example with actual path
./configure --with-llama=/home/user/llama.cpp

# With additional options
./configure \
    --with-llama=/home/user/llama.cpp \
    --prefix=/usr/local \
    --enable-shared
```

**Configure will:**
- Detect Tcl installation
- Locate OpenSSL
- Search for llama.cpp headers and libraries
- Generate Makefile from Makefile.in
- Create pkgIndex.tcl

**Verify configuration output:**

```bash
# Check for errors or warnings
grep -i "error\|warning" config.log

# Key variables should be detected
grep "LLAMA_INCLUDES\|LLAMA_LIBS" Makefile
```

### 7. Build Tcl-Llama

```bash
# Clean any previous builds
make clean

# Compile the extension
make

# Verify .so file was created
ls -lh tclllama.so
# Expected: ~134KB shared library
file tclllama.so
# Expected: ELF 64-bit LSB shared object
```

**Build Troubleshooting:**

If compilation fails:

```bash
# Check compiler flags
grep "COMPILE_FLAGS\|LINK_FLAGS" Makefile

# Review full build output
make V=1  # Verbose output

# Check for missing dependencies
ldd tclllama.so  # Lists required libraries
```

### 8. Install the Extension (Optional)

```bash
# Standard installation (requires sudo)
sudo make install

# Custom installation directory
make install DESTDIR=/opt/tclllama

# Verify installation
ls -lh /usr/local/lib/tclllama.so
# or wherever you installed it
```

### 9. Test the Build

#### Quick Verification

```bash
# Test loading the extension
tclsh << 'EOF'
load ./tclllama.so Tclllama
puts "Tclllama loaded successfully"
set version [llama version]
puts "Version: $version"
EOF
```

#### Run Example Script

Create `test.tcl`:

```tcl
#!/usr/bin/tclsh

load ./tclllama.so Tclllama

# Download a small model first (e.g., from huggingface)
set model "model.gguf"

if {![file exists $model]} {
    puts "Error: Model file $model not found"
    puts "Please download a GGUF model first"
    exit 1
}

llama init $model -n_ctx 512
puts "Model loaded successfully"

set result [llama generate "Hello, how are you?" -num_predict 50]
puts "Response: $result"

llama free
puts "Done"
```

Run:

```bash
tclsh test.tcl
```

## Building for Different Platforms

### Linux (x86_64)

Standard build as described above.

### Linux (ARM64/aarch64)

```bash
./configure --with-llama=/path/to/llama.cpp

# May need to specify ARM-specific options
CFLAGS="-O3 -march=native" make
```

### macOS

```bash
# Ensure Homebrew path is in PKG_CONFIG_PATH
export PKG_CONFIG_PATH="/usr/local/opt/openssl/lib/pkgconfig"

./configure --with-llama=/path/to/llama.cpp

make
```

### Windows (MSVC)

Using Visual Studio Developer Command Prompt:

```cmd
REM Set environment
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Configure
configure.bat --with-llama=C:\path\to\llama.cpp

REM Build
nmake
```

## Advanced Configuration

### Optimization Flags

For maximum performance:

```bash
./configure --with-llama=/path/to/llama.cpp \
    CFLAGS="-O3 -march=native -mtune=native" \
    CXXFLAGS="-O3 -march=native -mtune=native"

make
```

### Debug Build

For debugging and development:

```bash
./configure --with-llama=/path/to/llama.cpp \
    CFLAGS="-g -O0" \
    CXXFLAGS="-g -O0"

make
```

Then debug with:
```bash
gdb --args tclsh test.tcl
```

### Static Linking

To create standalone executable:

```bash
# Build llama.cpp with static libraries
cd llama.cpp/build
cmake .. -DBUILD_SHARED_LIBS=OFF
cmake --build . -j$(nproc)

cd ../../tcl-llama
./configure --with-llama=/path/to/llama.cpp \
    LDFLAGS="-static"
make
```

## Troubleshooting

### Issue: "llama.h: No such file or directory"

**Cause:** llama.cpp headers not found

**Solution:**
```bash
# Verify llama.cpp path
ls /path/to/llama.cpp/include/llama.h

# Check ggml.h location
find /path/to/llama.cpp -name "ggml.h"

# Reconfigure with correct path
./configure --with-llama=/path/to/llama.cpp
make clean && make
```

### Issue: "libllama.so: cannot open shared object file"

**Cause:** Runtime library path not set

**Solution:**
```bash
# Set library path before running
export LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH

# Or install to system library path
sudo cp /path/to/llama.cpp/build/bin/libllama.so* /usr/local/lib/
sudo ldconfig

# Test
ldd ./tclllama.so | grep libllama
```

### Issue: "cannot find -lcrypto"

**Cause:** OpenSSL development files not installed

**Solution:**
```bash
# Fedora/RHEL
dnf install openssl-devel

# Debian/Ubuntu
apt-get install libssl-dev

# macOS
brew install openssl
export PKG_CONFIG_PATH="/usr/local/opt/openssl/lib/pkgconfig"

# Reconfigure
./configure --with-llama=/path/to/llama.cpp
make clean && make
```

### Issue: "error: 'llama_sampler_chain_default_params' was not declared"

**Cause:** llama.cpp version mismatch

**Solution:**
```bash
# Update llama.cpp to latest
cd /path/to/llama.cpp
git pull origin master
cd build
cmake --build . -j$(nproc)

# Reconfigure tclllama
cd ../../tcl-llama
make clean
./configure --with-llama=/path/to/llama.cpp
make
```

### Issue: Segmentation fault during execution

**Cause:** Various possible issues (memory, corrupted model, etc.)

**Solution:**
```bash
# Build with debug symbols
make clean
./configure CFLAGS="-g -O0" CXXFLAGS="-g -O0" --with-llama=/path/to/llama.cpp
make

# Run under debugger
gdb -ex run -ex "bt" --args tclsh test.tcl

# Or use verbose mode
tclsh << 'EOF'
llama verbose 3
# ... your test code ...
EOF
```

## Build Variables Reference

```bash
# Common variables used during build

LLAMA_INCLUDES        # Path to llama.cpp headers
LLAMA_LIBS           # Linker flags for libllama.so
TCL_INCLUDE_SPEC     # Tcl include directory
TCL_LIB_SPEC         # Tcl library directory
CFLAGS               # C compiler flags
CXXFLAGS             # C++ compiler flags
LDFLAGS              # Linker flags
CC                   # C compiler (usually gcc)
CXX                  # C++ compiler (usually g++)
```

View actual values:
```bash
grep "^LLAMA_INCLUDES\|^LLAMA_LIBS\|^TCL_" Makefile
```

## Next Steps

After successful build:

1. **Test with Real Models**: Download a GGUF model and test inference
2. **Optimize for Your Hardware**: Adjust compilation flags for your CPU
3. **Integrate into Applications**: Use tclllama in your Tcl projects
4. **Contribute Improvements**: Submit patches back to the project

## Additional Resources

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Tcl Documentation**: https://www.tcl.tk/man/
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Model Downloads**: https://huggingface.co/models?library=gguf

---

**Version**: 1.0
**Last Updated**: December 2024
