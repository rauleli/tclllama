#!/usr/bin/env tclsh
# -----------------------------------------------------------------------------
# generation_benchmark.tcl -- Reproducible generative CPU measurements
#
# Purpose: measure cold model load, context creation, prefill, token generation,
#          chat, and one synchronous shared-round batch_step operation.
# Public interface: tclsh generation_benchmark.tcl package-dir model.gguf
#                    label ?repetitions? ?threads?
# Requirements: Tcl 8.6, tclllama 0.1.0, Linux sha256sum, compatible GGUF.
# Output: one TSV header plus cold and repeated operation records on stdout.
# Generation units are the median decoded-token count; requested_tokens records
# the cap. Throughput divides median units by median end-to-end duration.
# -----------------------------------------------------------------------------

namespace eval ::generationBenchmark {
    variable llamaRevision 1719dbb0ee7bdaf56de083079d3613bb670160a5
    variable threads 2
    variable contextSize 2048

    # Return median and median absolute deviation for a nonempty list of times.
    proc distribution {values} {
        set ordered [lsort -real $values]
        set count [llength $ordered]
        if {$count % 2} {
            set median [lindex $ordered [expr {$count / 2}]]
        } else {
            set median [expr {([lindex $ordered [expr {$count / 2 - 1}]] +
                [lindex $ordered [expr {$count / 2}]]) / 2.0}]
        }
        set deviations {}
        foreach value $ordered {lappend deviations [expr {abs($value - $median)}]}
        return [list $median [lindex [lsort -real $deviations] [expr {$count / 2}]]]
    }

    # Read Linux process RSS metrics when available; report NA on other systems.
    proc memory {} {
        set metrics [dict create rss_kb NA peak_rss_kb NA]
        if {![file readable /proc/self/status]} {return $metrics}
        set input [open /proc/self/status r]
        try {
            while {[gets $input line] >= 0} {
                if {[regexp {^VmRSS:\s+([0-9]+)\s+kB} $line -> value]} {
                    dict set metrics rss_kb $value
                } elseif {[regexp {^VmHWM:\s+([0-9]+)\s+kB} $line -> value]} {
                    dict set metrics peak_rss_kb $value
                }
            }
        } finally {
            close $input
        }
        return $metrics
    }

    # Emit a measurement row. The `kind` field separates setup from warm calls.
    proc emit {kind model hash operation samples units {requested NA}} {
        variable llamaRevision
        variable threads
        variable contextSize
        if {[llength $samples] == 0} {
            set median $samples
            set mad $samples
        } else {
            lassign [distribution $samples] median mad
        }
        set memory [::generationBenchmark::memory]
        set throughput NA
        if {[string is double -strict $median] && $median > 0 && [string is double -strict $units]} {
            set throughput [expr {1000.0 * $units / $median}]
        }
        puts [join [list $kind $model $hash $llamaRevision [info patchlevel] \
            [package provide tclllama] $threads $contextSize $operation $median $mad \
            $throughput $units [dict get $memory rss_kb] [dict get $memory peak_rss_kb] $requested] "\t"]
    }
}

if {$argc < 3 || $argc > 5} {
    puts stderr "Usage: generation_benchmark.tcl package-dir model.gguf label ?repetitions? ?threads?"
    exit 2
}
set packageDir [file normalize [lindex $argv 0]]
set modelPath [file normalize [lindex $argv 1]]
set label [lindex $argv 2]
set repetitions [expr {$argc >= 4 ? [lindex $argv 3] : 5}]
set threads [expr {$argc >= 5 ? [lindex $argv 4] : 2}]
foreach {name value minimum} [list repetitions $repetitions 3 threads $threads 1] {
    if {![string is integer -strict $value] || $value < $minimum} {
        puts stderr "$name must be an integer >= $minimum"
        exit 2
    }
}
if {![file isfile $modelPath]} {puts stderr "model not found: $modelPath"; exit 2}
lappend auto_path $packageDir
package require tclllama 0.1.0
set ::generationBenchmark::threads $threads
set hash [lindex [exec sha256sum -- $modelPath] 0]
set options [dict create n_threads $threads n_threads_batch $threads n_seq_max 1]
set modelStart [clock microseconds]
set model [llama::model_load $modelPath]
set modelMs [expr {([clock microseconds] - $modelStart) / 1000.0}]
set context {}
set batchContext {}
try {
    set contextStart [clock microseconds]
    set context [llama::context_create $model 2048 -options $options]
    set contextMs [expr {([clock microseconds] - $contextStart) / 1000.0}]
    set batchOptions [dict create n_threads $threads n_threads_batch $threads \
        n_seq_max 2 sequence_sampling true]
    set batchContextStart [clock microseconds]
    set batchContext [llama::context_create $model 2048 -options $batchOptions]
    set batchContextMs [expr {([clock microseconds] - $batchContextStart) / 1000.0}]
    llama::sequence $batchContext create 1
    set promptA [llama::tokenize $context {A deterministic prompt for a short CPU inference measurement.}]
    set promptB [llama::tokenize $batchContext {A second independent sequence for shared round sampling.}]
    set promptText {A deterministic prompt for a short CPU inference measurement.}
    set generationOptions [dict create temperature 0 top_k 1 seed 1234]
    set chatMessages [list [dict create role user content {Reply with one short word.}]]

    puts "record\tmodel\tsha256\tllama_revision\ttcl_version\ttclllama_version\tthreads\tn_ctx_requested\toperation\tmedian_ms\tmad_ms\tunits_per_second\tunits\trss_kb\tpeak_rss_kb\trequested_tokens"
    ::generationBenchmark::emit cold $label $hash model_load [list $modelMs] 1
    ::generationBenchmark::emit cold $label $hash context_create [list $contextMs] 1
    ::generationBenchmark::emit cold $label $hash batch_context_create [list $batchContextMs] 1

    set samples {}
    foreach _ [lrepeat $repetitions 1] {
        llama::clear_cache $context
        set start [clock microseconds]
        llama::prefill $context $promptA
        lappend samples [expr {([clock microseconds] - $start) / 1000.0}]
    }
    ::generationBenchmark::emit warm $label $hash prefill $samples [llength $promptA]

    set samples {}
    set counts {}
    foreach _ [lrepeat $repetitions 1] {
        llama::clear_cache $context
        set start [clock microseconds]
        set result [llama::generate $context $promptText -reset true -options $generationOptions -max_tokens 16 -details true]
        lappend samples [expr {([clock microseconds] - $start) / 1000.0}]
        lappend counts [dict get $result generated_tokens]
    }
    lassign [::generationBenchmark::distribution $counts] generated ignored
    ::generationBenchmark::emit warm $label $hash generate_16_end_to_end $samples $generated 16

    set samples {}
    set counts {}
    foreach _ [lrepeat $repetitions 1] {
        llama::clear_cache $context
        set start [clock microseconds]
        set result [llama::chat $context $chatMessages -options $generationOptions -max_tokens 16 -details true]
        lappend samples [expr {([clock microseconds] - $start) / 1000.0}]
        lappend counts [dict get $result generated_tokens]
    }
    lassign [::generationBenchmark::distribution $counts] generated ignored
    ::generationBenchmark::emit warm $label $hash chat_16 $samples $generated 16

    set samples {}
    foreach _ [lrepeat $repetitions 1] {
        llama::clear_cache $batchContext
        llama::batch_prefill $batchContext [list [list 0 $promptA] [list 1 $promptB]]
        set start [clock microseconds]
        llama::batch_step $batchContext {0 1}
        lappend samples [expr {([clock microseconds] - $start) / 1000.0}]
    }
    ::generationBenchmark::emit warm $label $hash batch_step_two_sequences $samples 2
} finally {
    if {$batchContext ne {}} {llama::context_free $batchContext}
    if {$context ne {}} {llama::context_free $context}
    llama::model_free $model
}
