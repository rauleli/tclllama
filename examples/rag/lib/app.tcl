# app.tcl -- Shared entry point for native and local Ollama RAG examples.
# Provides ::rag::app::run. Tcl 8.6, tclllama, rl_json, sha256 and the
# selected SQL driver/UDF are required. Input indices and models stay private.
package require Tcl 8.6
package require sha256
namespace eval ::rag::app {
    variable embed_context {}
    variable backend {}
    variable corpus {}
    variable store_config {}
    variable query_prefix {}
    variable min_score {}
}

# Retrieve verified evidence through the same code for both generators.
proc ::rag::app::retrieve {question topk} {
    variable backend
    variable corpus
    variable store_config
    variable embed_context
    variable query_prefix
    variable min_score
    return [::rag::retrieve::query $backend $corpus $store_config $embed_context $query_prefix $min_score $question $topk]
}

# Run the interactive application with explicit paths and settings. Arguments:
# backend location database user extension corpus embedding_model config_file
# generator. For SQLite, location is the database path, extension the SQLite
# extension and database/user are '-'. For MySQL, location is a local socket,
# database/user identify its private database and extension is '-'. Generator
# is a GGUF path for native mode or an installed Ollama model for ollama mode.
# config_file is a Tcl dictionary (data, never evaluated) containing
# model_sha256, dimension, document_prefix and query_prefix. It must be exactly
# the configuration used for ingestion. Resources are released on return.
proc ::rag::app::run {mode args} {
    variable embed_context
    variable backend
    variable corpus
    variable store_config
    variable query_prefix
    variable min_score
    if {$mode ni {native ollama} || [llength $args] != 9} {error "usage: backend location database user extension corpus embedding_model config_file generator"}
    lassign $args backend location database user extension corpus embedding config_file generator
    if {$backend ni {sqlite mysql}} {error "backend must be sqlite or mysql"}
    set f [open $config_file r]
    try {fconfigure $f -encoding utf-8; set config [string trim [read $f]]} finally {close $f}
    foreach key {model_sha256 dimension embedding_context document_prefix query_prefix min_score topk max_context_chars max_history_chars generation_max_tokens} {
        if {![dict exists $config $key]} {error "configuration missing $key"}
    }
    ::rag::binding::verify $config
    set expected_hash [dict get $config model_sha256]
    if {![regexp {^[0-9a-f]{64}$} $expected_hash] || [::sha2::sha256 -hex -filename $embedding] ne $expected_hash} {error "embedding model hash mismatch"}
    set dimension [dict get $config dimension]
    set query_prefix [dict get $config query_prefix]
    set min_score [dict get $config min_score]
    if {[info exists ::env(TCLLLAMA_RAG_MIN_SCORE)]} {set min_score $::env(TCLLLAMA_RAG_MIN_SCORE)}
    set topk [dict get $config topk]
    if {[info exists ::env(TCLLLAMA_RAG_TOPK)]} {set topk $::env(TCLLLAMA_RAG_TOPK)}
    set threads [expr {[dict exists $config threads] ? [dict get $config threads] : 2}]
    if {![string is integer -strict $threads] || $threads < 1 || $threads > 256} {error "invalid thread count"}
    set store_config [list rag-ingest-v1 $config prefix [dict get $config document_prefix] dimension $dimension]
    set store_open 0
    set model {}
    set embed_context {}
    set generation_context {}
    try {
        if {$backend eq "sqlite"} {
            ::rag::sqlite::open $location 0
            set store_open 1
            set extension [file normalize $extension]
            ::rag::sqlite::db enable_load_extension 1
            try {::rag::sqlite::db eval {SELECT load_extension($extension,'sqlite3_similarity_init')}} finally {::rag::sqlite::db enable_load_extension 0}
        } else {
            ::rag::mysql::open $location $database $user
            set store_open 1
        }
        ::rag::${backend}::check_config $corpus $store_config
        set model [llama::model_load $embedding]
        set embed_context [llama::embedding_context_create $model [dict get $config embedding_context] -options [list pooling_type model attention_type model n_threads $threads n_threads_batch $threads n_batch 256 n_ubatch 256 n_seq_max 4]]
        if {[dict get [llama::info $embed_context] model n_embd_out] != $dimension} {error "model dimension mismatch"}
        if {$mode eq "native"} {
            set generation_context [llama::init $generator 2048 -options {n_threads 2 n_threads_batch 2}]
            set generate [list ::rag::generate::native $generation_context {temperature 0.2 top_k 40 top_p 0.9} [dict get $config generation_max_tokens]]
        } else {
            set port 11434
            if {[info exists ::env(TCLLLAMA_OLLAMA_PORT)]} {set port $::env(TCLLLAMA_OLLAMA_PORT)}
            set generate [list ::rag::generate::ollama $port $generator 120000 [dict get $config generation_max_tokens]]
        }
        set model_label $generator
        if {[info exists ::env(TCLLLAMA_RAG_MODEL_LABEL)]} {
            set model_label $::env(TCLLLAMA_RAG_MODEL_LABEL)
        } elseif {$mode eq "native"} {
            set model_label [file tail $generator]
        }
        set backend_label [string totitle $backend]
        puts "Chat RAG local. Escribe /ayuda para ver los comandos."
        puts "Modelo seleccionado: $model_label"
        puts "Base seleccionada: $backend_label ($corpus)"
        puts "Cada pregunta se procesa de forma independiente."
        ::rag::repl::run stdin stdout ::rag::app::retrieve $generate $topk [dict get $config max_context_chars] [dict get $config max_history_chars]
    } finally {
        if {$generation_context ne ""} {catch {llama::free $generation_context}}
        if {$embed_context ne ""} {catch {llama::context_free $embed_context}}
        if {$model ne ""} {catch {llama::model_free $model}}
        if {$store_open} {catch {::rag::${backend}::close}}
    }
}
