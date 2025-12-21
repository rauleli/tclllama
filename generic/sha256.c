// sha256.c - Tcl binding mejorado para OpenSSL SHA256
// Mejoras: Registry de contextos, mejor manejo de memoria, validación robusta v2.0
#include <tcl.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================
// REGISTRY DE CONTEXTOS (thread-safe para single thread)
// ============================================

#define MAX_CONTEXTS 256

typedef struct {
    EVP_MD_CTX *ctx;
    int in_use;
    unsigned long id;
} SHA256Context;

static SHA256Context g_contexts[MAX_CONTEXTS];
static unsigned long g_next_id = 1;
static int g_initialized = 0;

static void init_context_registry(void) {
    if (!g_initialized) {
        memset(g_contexts, 0, sizeof(g_contexts));
        g_initialized = 1;
    }
}

static unsigned long register_context(EVP_MD_CTX *ctx) {
    init_context_registry();
    
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (!g_contexts[i].in_use) {
            g_contexts[i].ctx = ctx;
            g_contexts[i].in_use = 1;
            g_contexts[i].id = g_next_id++;
            return g_contexts[i].id;
        }
    }
    
    return 0; // No space available
}

static EVP_MD_CTX* get_context(unsigned long id) {
    init_context_registry();
    
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (g_contexts[i].in_use && g_contexts[i].id == id) {
            return g_contexts[i].ctx;
        }
    }
    
    return NULL;
}

static int unregister_context(unsigned long id) {
    init_context_registry();
    
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (g_contexts[i].in_use && g_contexts[i].id == id) {
            if (g_contexts[i].ctx) {
                EVP_MD_CTX_free(g_contexts[i].ctx);
                g_contexts[i].ctx = NULL;
            }
            g_contexts[i].in_use = 0;
            g_contexts[i].id = 0;
            return 1;
        }
    }
    
    return 0;
}

// ============================================
// sha256::file - Hash a file directly
// ============================================
static int Sha256File_Cmd(ClientData clientData, Tcl_Interp *interp,
                          int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "filename");
        return TCL_ERROR;
    }

    const char *filename = Tcl_GetString(objv[1]);
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        Tcl_SetObjResult(interp, Tcl_ObjPrintf("Cannot open file: %s", filename));
        return TCL_ERROR;
    }

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        fclose(fp);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create digest context", -1));
        return TCL_ERROR;
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1) {
        EVP_MD_CTX_free(ctx);
        fclose(fp);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to initialize digest", -1));
        return TCL_ERROR;
    }

    unsigned char buffer[65536];
    size_t bytes;

    while ((bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (EVP_DigestUpdate(ctx, buffer, bytes) != 1) {
            EVP_MD_CTX_free(ctx);
            fclose(fp);
            Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to update digest", -1));
            return TCL_ERROR;
        }
    }

    if (ferror(fp)) {
        EVP_MD_CTX_free(ctx);
        fclose(fp);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Error reading file", -1));
        return TCL_ERROR;
    }

    fclose(fp);

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len;

    if (EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to finalize digest", -1));
        return TCL_ERROR;
    }

    EVP_MD_CTX_free(ctx);

    char hex[65];
    for (unsigned int i = 0; i < hash_len; i++) {
        sprintf(hex + (i * 2), "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';

    Tcl_SetObjResult(interp, Tcl_NewStringObj(hex, -1));
    return TCL_OK;
}

// ============================================
// sha256::init - Create a new streaming context
// ============================================
static int Sha256Init_Cmd(ClientData clientData, Tcl_Interp *interp,
                          int objc, Tcl_Obj *const objv[]) {
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create context", -1));
        return TCL_ERROR;
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1) {
        EVP_MD_CTX_free(ctx);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to initialize digest", -1));
        return TCL_ERROR;
    }

    // Register context and return ID
    unsigned long id = register_context(ctx);
    if (id == 0) {
        EVP_MD_CTX_free(ctx);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Too many contexts", -1));
        return TCL_ERROR;
    }

    char handle[32];
    snprintf(handle, sizeof(handle), "sha256ctx_%lu", id);
    Tcl_SetObjResult(interp, Tcl_NewStringObj(handle, -1));
    return TCL_OK;
}

// ============================================
// sha256::update - Feed data to context
// ============================================
static int Sha256Update_Cmd(ClientData clientData, Tcl_Interp *interp,
                            int objc, Tcl_Obj *const objv[]) {
    if (objc != 3) {
        Tcl_WrongNumArgs(interp, 1, objv, "handle data");
        return TCL_ERROR;
    }

    const char *handle = Tcl_GetString(objv[1]);
    unsigned long id;
    if (sscanf(handle, "sha256ctx_%lu", &id) != 1) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }

    EVP_MD_CTX *ctx = get_context(id);
    if (!ctx) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Context not found", -1));
        return TCL_ERROR;
    }

    int data_len;
    const unsigned char *data = Tcl_GetByteArrayFromObj(objv[2], &data_len);

    if (EVP_DigestUpdate(ctx, data, data_len) != 1) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to update digest", -1));
        return TCL_ERROR;
    }

    return TCL_OK;
}

// ============================================
// sha256::final - Finalize and get hash
// ============================================
static int Sha256Final_Cmd(ClientData clientData, Tcl_Interp *interp,
                           int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "handle");
        return TCL_ERROR;
    }

    const char *handle = Tcl_GetString(objv[1]);
    unsigned long id;
    if (sscanf(handle, "sha256ctx_%lu", &id) != 1) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Invalid handle", -1));
        return TCL_ERROR;
    }

    EVP_MD_CTX *ctx = get_context(id);
    if (!ctx) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Context not found", -1));
        return TCL_ERROR;
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;

    if (EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        unregister_context(id);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to finalize digest", -1));
        return TCL_ERROR;
    }

    // Unregister context (frees memory)
    unregister_context(id);

    char hex[65];
    for (unsigned int i = 0; i < hash_len; i++) {
        sprintf(hex + (i * 2), "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';

    Tcl_SetObjResult(interp, Tcl_NewStringObj(hex, -1));
    return TCL_OK;
}

// ============================================
// sha256::data - Hash raw data directly
// ============================================
static int Sha256Data_Cmd(ClientData clientData, Tcl_Interp *interp,
                          int objc, Tcl_Obj *const objv[]) {
    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "data");
        return TCL_ERROR;
    }

    int data_len;
    const unsigned char *data = Tcl_GetByteArrayFromObj(objv[1], &data_len);

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len;

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Failed to create context", -1));
        return TCL_ERROR;
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1 ||
        EVP_DigestUpdate(ctx, data, data_len) != 1 ||
        EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Digest failed", -1));
        return TCL_ERROR;
    }

    EVP_MD_CTX_free(ctx);

    char hex[65];
    for (unsigned int i = 0; i < hash_len; i++) {
        sprintf(hex + (i * 2), "%02x", hash[i]);
    }
    hex[hash_len * 2] = '\0';

    Tcl_SetObjResult(interp, Tcl_NewStringObj(hex, -1));
    return TCL_OK;
}

// ============================================
// sha256::cleanup - Manual cleanup of all contexts
// ============================================
static int Sha256Cleanup_Cmd(ClientData clientData, Tcl_Interp *interp,
                             int objc, Tcl_Obj *const objv[]) {
    if (objc != 1) {
        Tcl_WrongNumArgs(interp, 1, objv, "");
        return TCL_ERROR;
    }

    init_context_registry();

    int freed = 0;
    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (g_contexts[i].in_use) {
            if (g_contexts[i].ctx) {
                EVP_MD_CTX_free(g_contexts[i].ctx);
                g_contexts[i].ctx = NULL;
            }
            g_contexts[i].in_use = 0;
            g_contexts[i].id = 0;
            freed++;
        }
    }

    Tcl_SetObjResult(interp, Tcl_ObjPrintf("Freed %d contexts", freed));
    return TCL_OK;
}

// ============================================
// Package initialization
// ============================================
int Sha256_Init(Tcl_Interp *interp) {
    if (Tcl_InitStubs(interp, "8.6", 0) == NULL) {
        return TCL_ERROR;
    }

    // Initialize context registry
    init_context_registry();

    // Create sha256 namespace
    Tcl_CreateNamespace(interp, "::sha256", NULL, NULL);

    // Register commands
    Tcl_CreateObjCommand(interp, "::sha256::file", Sha256File_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::sha256::init", Sha256Init_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::sha256::update", Sha256Update_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::sha256::final", Sha256Final_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::sha256::data", Sha256Data_Cmd, NULL, NULL);
    Tcl_CreateObjCommand(interp, "::sha256::cleanup", Sha256Cleanup_Cmd, NULL, NULL);

    // Keep old command for compatibility
    Tcl_CreateObjCommand(interp, "sha256file", Sha256File_Cmd, NULL, NULL);

    Tcl_PkgProvide(interp, "sha256", "2.0");
    return TCL_OK;
}
