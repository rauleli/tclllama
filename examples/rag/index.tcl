#!/usr/bin/env tclsh
# index.tcl -- Create or resume a private RAG corpus with tclllama embeddings.
# Requires Tcl 8.6, tclllama, rl_json, json::write, sha256 and one SQL driver.
# See README.md for the experimental contract and explicit configuration fields.
# No model downloads occur; input and database files remain outside the checkout.
package require Tcl 8.6
package require tclllama 0.1.0
package require sha256
package require rl_json 0.15.7
set here [file dirname [file normalize [info script]]]
if {[llength $argv] != 10} {
    error "usage: backend location database user create|resume corpus source_id input_jsonl config_file embedding_model"
}
lassign $argv backend location database user action corpus source_id input_file config_file embedding
if {$backend ni {sqlite mysql} || $action ni {create resume}} {error "invalid backend or action"}
source [file join $here lib ${backend}_store.tcl]
foreach name {binding fragment extract ingest} {source [file join $here lib $name.tcl]}
set config_channel [open $config_file r]
try {fconfigure $config_channel -encoding utf-8; set config [string trim [read $config_channel]]} finally {close $config_channel}
foreach key {model_sha256 model_reference model_revision quantization pooling normalization input_sha256 dimension embedding_context document_prefix token_budget batch_size} {
    if {![dict exists $config $key]} {error "configuration missing $key"}
}
if {[dict get $config pooling] ne "model" || [dict get $config normalization] ne "l2"} {error "unsupported pooling or normalization"}
foreach key {model_reference model_revision quantization} {
    if {[dict get $config $key] eq ""} {error "empty model metadata: $key"}
}
::rag::binding::verify $config
set model_sha [::sha2::sha256 -hex -filename $embedding]
set input_sha [::sha2::sha256 -hex -filename $input_file]
if {$model_sha ne [dict get $config model_sha256] || $input_sha ne [dict get $config input_sha256]} {error "input or model hash mismatch"}
set dimension [dict get $config dimension]
set prefix [dict get $config document_prefix]
set budget [dict get $config token_budget]
set batch_size [dict get $config batch_size]
set threads [expr {[dict exists $config threads] ? [dict get $config threads] : 2}]
if {![string is integer -strict $threads] || $threads < 1 || $threads > 256} {error "invalid thread count"}
if {![string is integer -strict $batch_size] || $batch_size < 1 || $batch_size > 4 || ![string is integer -strict $budget] || $budget < 1 || $budget > 256} {error "invalid token budget or batch size"}
set opened 0
set model {}
set context {}
try {
    set model [llama::model_load $embedding]
    set context [llama::embedding_context_create $model [dict get $config embedding_context] -options [list pooling_type model attention_type model n_threads $threads n_threads_batch $threads n_batch 1024 n_ubatch 1024 n_seq_max 4]]
    if {[dict get [llama::info $context] model n_embd_out] != $dimension} {error "model dimension mismatch"}
    set count_command [list ::rag::fragment::llama_count $context]
    set records {}
    set seen {}
    set lines 0
    set input [open $input_file r]
    try {
        fconfigure $input -encoding utf-8 -translation lf
        while {[gets $input line] >= 0} {
            incr lines
            set parts [::rag::extract::record $line $source_id $input_sha $prefix $budget $count_command]
            set key [list [::rl_json::json get $line source] [::rl_json::json get $line conversation_id] [::rl_json::json get $line node_id] [::rl_json::json get $line part]]
            if {[dict exists $seen $key]} {error "duplicate input key at line $lines"}
            dict set seen $key 1
            lappend records {*}$parts
            if {$lines % 500 == 0} {puts stderr "fragmented_records=$lines fragments=[llength $records]"; flush stderr}
        }
    } finally {close $input}
    if {[::sha2::sha256 -hex -filename $input_file] ne $input_sha} {error "input changed during fragmentation"}
    if {$backend eq "sqlite"} {
        ::rag::sqlite::open $location [expr {$action eq "create"}]
        set opened 1
    } else {
        ::rag::mysql::open $location $database $user
        set opened 1
        if {$action eq "create"} {::rag::mysql::init}
    }
    set count [::rag::ingest::run $backend $corpus $config $dimension $records $context $prefix $batch_size]
    puts "input_records=$lines fragments=$count status=complete"
} finally {
    if {$opened} {catch {::rag::${backend}::close}}
    if {$context ne ""} {catch {llama::context_free $context}}
    if {$model ne ""} {catch {llama::model_free $model}}
}
