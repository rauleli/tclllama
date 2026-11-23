#!/bin/bash -norc
dnl Configure tclllama against Tcl and a reproducible PIC build of llama.cpp.

AC_INIT([tclllama], [7.5])
m4_include([tclconfig/tcl.m4])
TEA_INIT([3.13])
AC_CONFIG_AUX_DIR(tclconfig)

TEA_PATH_TCLCONFIG
TEA_LOAD_TCLCONFIG

AC_PROG_CXX
CC=$CXX

if test "${GCC}" = "yes"; then
    TEA_ADD_CFLAGS([-std=c++11 -O3])
else
    TEA_ADD_CFLAGS([-std=c++11])
fi

AC_ARG_WITH([llama],
    [AS_HELP_STRING([--with-llama=DIR], [llama.cpp source tree])],
    [llama_path="$withval"],
    [AC_MSG_ERROR([Especifique --with-llama=/ruta/a/llama.cpp])])

AC_ARG_WITH([llama-build],
    [AS_HELP_STRING([--with-llama-build=DIR], [llama.cpp static PIC build directory])],
    [llama_build_path="$withval"],
    [llama_build_path="$llama_path/build-1719dbb-static-pic"])

LLAMA_INCLUDES="-I$llama_path/include -I$llama_path/ggml/include -I$llama_path/common -I$llama_path/vendor"
LLAMA_DIR="$llama_path"
LLAMA_ARCHIVES="$llama_build_path/src/libllama.a $llama_build_path/ggml/src/libggml.a $llama_build_path/ggml/src/libggml-cpu.a $llama_build_path/ggml/src/libggml-base.a"
for archive in $LLAMA_ARCHIVES; do
    if test ! -f "$archive"; then
        AC_MSG_ERROR([Falta $archive; se requiere una build estática PIC de llama.cpp])
    fi
done
AC_MSG_NOTICE([llama.cpp source: $llama_path])
AC_MSG_NOTICE([llama.cpp static build: $llama_build_path])

AC_CHECK_LIB([crypto], [EVP_DigestInit], [],
    [AC_MSG_ERROR([OpenSSL libcrypto not found. Instala openssl-devel])])

TEA_PUBLIC_TCL_HEADERS
TEA_ENABLE_THREADS
TEA_ENABLE_SHARED
if test "${SHARED_BUILD}" != "1"; then
    AC_MSG_ERROR([tclllama requiere una build shared para package require])
fi
TEA_CONFIG_CFLAGS
TEA_MAKE_LIB
TEA_PROG_TCLSH
AC_SUBST([CFLAGS_DEFAULT])
AC_SUBST([CFLAGS_WARNING])
AC_SUBST([SHLIB_CFLAGS])

AC_SUBST([LLAMA_INCLUDES])
AC_SUBST([LLAMA_DIR])
AC_SUBST([LLAMA_ARCHIVES])
AC_SUBST([TCL_INCLUDE_SPEC])
AC_SUBST([TCL_SHLIB_SUFFIX])
AC_SUBST([TCL_LIB_SPEC])
AC_SUBST([TCL_STUB_LIB_SPEC])

AC_CONFIG_FILES([Makefile pkgIndex.tcl])
AC_OUTPUT
