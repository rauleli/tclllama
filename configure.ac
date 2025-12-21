#!/bin/bash -norc
dnl        configure.ac v5.0
dnl        Soporte para --with-llama y detección dinámica de paths

AC_INIT([tclllama], [1.0])

# --- TEA SETUP ---
TEA_INIT([3.9])

AC_CONFIG_AUX_DIR(tclconfig)

# --- LOAD TCL CONFIG ---
TEA_PATH_TCLCONFIG
TEA_LOAD_TCLCONFIG

# --- SYSTEM SETUP ---
TEA_SYSTEM_INFO

# --- SOURCES ---
TEA_ADD_SOURCES([tclllama.c sha256.c])

# --- C++ COMPILER SETUP ---
AC_PROG_CXX
CC=$CXX

# --- FLAGS OPTIMIZACION ---
if test "${GCC}" = "yes" ; then
    TEA_ADD_CFLAGS([-std=c++11 -O3])
else
    TEA_ADD_CFLAGS([-std=c++11])
fi

# ======================================================================
#  ARGUMENTO --with-llama
#  Permite al usuario definir: ./configure --with-llama=/ruta/a/llama
# ======================================================================

AC_ARG_WITH([llama],
    [AS_HELP_STRING([--with-llama],
       [directorio raíz de llama.cpp (busca headers en include/ o raíz, libs en lib/ o raíz)])],
    [
        llama_path="$withval"
        AC_MSG_NOTICE([Usando llama.cpp en: $llama_path])

        # 1. Configurar Headers (buscamos en root, include, y ggml/include)
        # Esto cubre builds caseros y instalaciones tipo /usr/local
        LLAMA_INCLUDES="-I\"$llama_path\" -I\"$llama_path/include\" -I\"$llama_path/ggml/include\""
        TEA_ADD_INCLUDES([-I"$llama_path" -I"$llama_path/include" -I"$llama_path/ggml/include"])

        # 2. Configurar Librerías (buscamos en root, lib, build y build/bin)
        # Esto cubre si compilaste en la raiz, en build/, o en build/bin/
        if test -f "$llama_path/libllama.so"; then
             LLAMA_LIBS="-L\"$llama_path\" -lllama"
             TEA_ADD_LIBS([-L"$llama_path" -lllama])
        elif test -f "$llama_path/build/bin/libllama.so"; then
             LLAMA_LIBS="-L\"$llama_path/build/bin\" -lllama"
             TEA_ADD_LIBS([-L"$llama_path/build/bin" -lllama])
        elif test -f "$llama_path/build/libllama.so"; then
             LLAMA_LIBS="-L\"$llama_path/build\" -lllama"
             TEA_ADD_LIBS([-L"$llama_path/build" -lllama])
        elif test -f "$llama_path/lib/libllama.so"; then
             LLAMA_LIBS="-L\"$llama_path/lib\" -lllama"
             TEA_ADD_LIBS([-L"$llama_path/lib" -lllama])
        else
             # Si no lo encontramos, confiamos en que el linker lo hallará o tronará
             AC_MSG_WARN([No se encontró libllama.so en rutas estándar dentro de $llama_path. Asumiendo path manual.])
             LLAMA_LIBS="-L\"$llama_path\" -lllama"
             TEA_ADD_LIBS([-L"$llama_path" -lllama])
        fi
    ],
    [
        # Comportamiento por defecto si NO se usa --with-llama
        AC_MSG_NOTICE([No se especificó --with-llama. Asumiendo instalación del sistema o local.])
        LLAMA_INCLUDES="-I."
        LLAMA_LIBS="-L. -lllama"
        TEA_ADD_INCLUDES([-I.])
        TEA_ADD_LIBS([-L. -lllama])
    ]
)

# ======================================================================
#  OPENSSL (Sistema)
# ======================================================================
AC_CHECK_LIB(crypto, EVP_DigestInit, [
    TEA_ADD_LIBS([-lcrypto])
], [
    AC_MSG_ERROR([OpenSSL libcrypto not found. Instala libssl-dev / openssl-devel])
])

# --- TEA BOILERPLATE ---
TEA_PUBLIC_TCL_HEADERS
TEA_ENABLE_THREADS
TEA_ENABLE_SHARED
TEA_CONFIG_CFLAGS
TEA_MAKE_LIB
TEA_PROG_TCLSH

# --- EXPORT SUBSTITUTIONS ---
AC_SUBST(LLAMA_INCLUDES)
AC_SUBST(LLAMA_LIBS)
AC_SUBST(TCL_INCLUDE_SPEC)
AC_SUBST(TCL_SHLIB_SUFFIX)
AC_SUBST(TCL_LIB_SPEC)
AC_SUBST(TCL_STUB_LIB_SPEC)

# --- OUTPUT ---
AC_CONFIG_FILES([Makefile pkgIndex.tcl])
AC_OUTPUT
