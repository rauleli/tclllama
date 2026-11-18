#!/usr/bin/env tclsh
# -----------------------------------------------------------------------------
# embeddings_benchmark.tcl — Reproducible CPU embedding measurements
#
# Purpose: compare end-to-end individual calls with native multi-text batches
#          for fixed text sets and report cold/setup costs and warm dispersion.
# Public entry point: Tcl script arguments, TSV written to stdout.
# Requirements: Tcl 8.6, tclllama 7.5, sha256sum, /proc, and a compatible GGUF.
# Usage: tclsh tests/embeddings_benchmark.tcl package-dir model.gguf label
#        ?repetitions? ?task-prefix? ?threads?
# -----------------------------------------------------------------------------

if {$argc < 3 || $argc > 6} {
    puts stderr "Usage: embeddings_benchmark.tcl packageDir modelPath modelLabel ?repetitions? ?taskPrefix? ?threads?"
    exit 2
}

set packageDir [file normalize [lindex $argv 0]]
set modelPath [file normalize [lindex $argv 1]]
set modelLabel [lindex $argv 2]
set repetitions [expr {$argc >= 4 ? [lindex $argv 3] : 7}]
set taskPrefix [expr {$argc >= 5 ? [lindex $argv 4] : {}}]
set threads [expr {$argc >= 6 ? [lindex $argv 5] : 2}]
if {![string is integer -strict $repetitions] || $repetitions < 5} {
    puts stderr "repetitions must be an integer of at least 5"
    exit 2
}
if {![string is integer -strict $threads] || $threads < 1} {
    puts stderr "threads must be a positive integer"
    exit 2
}
if {![file isfile $modelPath]} {
    puts stderr "model file is not readable: $modelPath"
    exit 2
}

package require tcltest 2
lappend auto_path $packageDir
package require tclllama 7.5

namespace eval ::embeddingBenchmark {}

# Read a CPU model name without requiring a shell pipeline or extra package.
proc ::embeddingBenchmark::cpuName {} {
    if {![file readable /proc/cpuinfo]} {return unavailable}
    set channel [open /proc/cpuinfo r]
    try {
        while {[gets $channel line] >= 0} {
            if {[regexp {^model name\s*:\s*(.*)$} $line -> value]} {
                return [string trim $value]
            }
        }
    } finally {
        close $channel
    }
    return unavailable
}

# Return current resident and peak resident memory in KiB where Linux reports it.
proc ::embeddingBenchmark::memoryStatus {} {
    set values [dict create rss_kb NA peak_rss_kb NA]
    if {![file readable /proc/self/status]} {return $values}
    set channel [open /proc/self/status r]
    try {
        while {[gets $channel line] >= 0} {
            if {[regexp {^VmRSS:\s+([0-9]+)\s+kB} $line -> value]} {
                dict set values rss_kb $value
            } elseif {[regexp {^VmHWM:\s+([0-9]+)\s+kB} $line -> value]} {
                dict set values peak_rss_kb $value
            }
        }
    } finally {
        close $channel
    }
    return $values
}

# Return sorted numeric median and median absolute deviation for measured times.
proc ::embeddingBenchmark::distribution {samples} {
    set ordered [lsort -real $samples]
    set count [llength $ordered]
    if {$count % 2} {
        set median [lindex $ordered [expr {$count / 2}]]
    } else {
        set median [expr {([lindex $ordered [expr {$count / 2 - 1}]] +
            [lindex $ordered [expr {$count / 2}]]) / 2.0}]
    }
    set deviations {}
    foreach value $ordered {lappend deviations [expr {abs($value - $median)}]}
    set mad [lindex [lsort -real $deviations] [expr {$count / 2}]]
    return [dict create median_ms $median mad_ms $mad min_ms [lindex $ordered 0] \
        max_ms [lindex $ordered end]]
}

# Remove tabs/newlines so each measured record stays one TSV row.
proc ::embeddingBenchmark::field {value} {
    return [string map [list "\t" { } "\r" { } "\n" { }] $value]
}

set texts [list \
    {A short English sentence about language models.} \
    {Texto breve en español con caracteres: acción, pingüino y café.} \
    {A longer document describes native inference and vector representations. Batches should preserve order, dimensions, and independent sequence pooling.} \
    {search query: ¿Cómo se representan varias secuencias en un mismo batch de inferencia?} \
    {This fifth example repeats a few important words about models, vectors, language, and independent context reuse.} \
    {El sexto texto combina inglés y español para ejercitar UTF-8: naïve, 東京, niño, y mañana.} \
    {A related description of native batch evaluation, pooling by sequence identifier, and Float32 vector output.} \
    {Documento final: una entrada diferente, de longitud media, para comparar el orden de resultados.}]
if {$taskPrefix ne {}} {
    set prefixed {}
    foreach text $texts {lappend prefixed "$taskPrefix$text"}
    set texts $prefixed
}

set modelLoadStart [clock microseconds]
set model [llama::model_load $modelPath]
set modelLoadMs [expr {([clock microseconds] - $modelLoadStart) / 1000.0}]
set context {}
try {
    set contextCreateStart [clock microseconds]
    set contextOptions [dict create \
        pooling_type model attention_type model \
        n_batch 2048 n_ubatch 512 n_seq_max 8 \
        n_threads $threads n_threads_batch $threads]
    set context [llama::embedding_context_create $model -options $contextOptions]
    set contextCreateMs [expr {([clock microseconds] - $contextCreateStart) / 1000.0}]
    set info [llama::info $context]
    set effective [dict get $info context effective]
    set dimension [dict get $info model n_embd_out]
    set modelHash [lindex [exec sha256sum -- $modelPath] 0]
    set engine [llama::engine_info]
    set engineVersion [dict get $engine version]
    set cpu [::embeddingBenchmark::cpuName]
    set backendInfo [llama::backend_info]
    set backendNames {}
    foreach backend [dict get $backendInfo backends] {
        lappend backendNames [dict get $backend name]
    }
    set systemInfo [dict get $engine system_info]
    set memory [::embeddingBenchmark::memoryStatus]

    # Token counts are reported independently; timing llama::tokenize here does
    # not split tokenize/evaluate/extract inside the public end-to-end commands.
    set tokenCounts {}
    set totalTokens 0
    set tokenizationStart [clock microseconds]
    foreach text $texts {
        set count [llength [llama::tokenize $context $text]]
        lappend tokenCounts $count
        incr totalTokens $count
    }
    set tokenizationMs [expr {([clock microseconds] - $tokenizationStart) / 1000.0}]

    puts "record\tmodel\tmodel_blob\tsha256\tllama_revision\tllama_version\tengine_system_info\ttcl_version\ttclllama_version\tcpu\tbackend\tdimension\tn_ctx_requested\tn_ctx_effective\tn_ctx_seq\tn_batch\tn_ubatch\tn_seq_max\tthreads\tpooling\tnormalization\tformat\ttexts\ttokens\trepetitions\tmodel_load_ms\tcontext_create_ms\ttoken_count_probe_ms\tfirst_run_ms\tmedian_ms\tmad_ms\tmin_ms\tmax_ms\tms_per_vector\tvectors_per_second\ttokens_per_second\tspeedup_vs_individual\trss_kb\tpeak_rss_kb\tnative_calls_per_repetition"

    set common [list \
        [::embeddingBenchmark::field $modelLabel] \
        [file tail $modelPath] $modelHash \
        1719dbb0ee7bdaf56de083079d3613bb670160a5 $engineVersion \
        [::embeddingBenchmark::field $systemInfo] [info patchlevel] \
        [package provide tclllama] \
        [::embeddingBenchmark::field $cpu] [join $backendNames ,] $dimension \
        [dict get [dict get $info context requested] n_ctx] \
        [dict get $effective n_ctx] [dict get $effective n_ctx_seq] \
        [dict get $effective n_batch] [dict get $effective n_ubatch] \
        [dict get $effective n_seq_max] [dict get $effective n_threads] \
        [dict get $effective pooling_type] none list]

    # Emit setup rows separately from model inference so load costs are not
    # folded into latency or throughput.
    foreach record [list [list model_load $modelLoadMs] [list context_create $contextCreateMs] \
                         [list token_count_probe $tokenizationMs]] {
        lassign $record kind duration
        set loadValue [expr {$kind eq "model_load" ? $duration : "NA"}]
        set contextValue [expr {$kind eq "context_create" ? $duration : "NA"}]
        set tokenValue [expr {$kind eq "token_count_probe" ? $duration : "NA"}]
        puts [join [concat [list $kind] $common [list 0 $totalTokens 0 $loadValue \
            $contextValue $tokenValue NA NA NA NA NA NA NA NA NA \
            [dict get $memory rss_kb] [dict get $memory peak_rss_kb] 0]] "\t"]
    }

    # The first individual call is kept as a cold/first-evaluation observation.
    set coldStart [clock microseconds]
    llama::embed $context [lindex $texts 0]
    set coldMs [expr {([clock microseconds] - $coldStart) / 1000.0}]
    set memory [::embeddingBenchmark::memoryStatus]
    puts [join [concat [list cold_individual] $common [list 1 [lindex $tokenCounts 0] 0 \
        $modelLoadMs $contextCreateMs $tokenizationMs $coldMs NA NA NA NA NA NA NA NA \
        [dict get $memory rss_kb] [dict get $memory peak_rss_kb] 1]] "\t"]

    foreach count {2 4 8} {
        set group [lrange $texts 0 [expr {$count - 1}]]
        set groupTokens [lrange $tokenCounts 0 [expr {$count - 1}]]
        set tokenTotal 0
        foreach value $groupTokens {incr tokenTotal $value}

        # First call for each batch shape is reported separately from steady
        # repetitions because it can include graph reservation/reconfiguration.
        set firstStart [clock microseconds]
        llama::embed_batch $context $group
        set firstMs [expr {([clock microseconds] - $firstStart) / 1000.0}]
        llama::embed_batch $context $group
        set batchSamples {}
        set individualSamples {}
        for {set repetition 0} {$repetition < $repetitions} {incr repetition} {
            set start [clock microseconds]
            llama::embed_batch $context $group
            lappend batchSamples [expr {([clock microseconds] - $start) / 1000.0}]

            set start [clock microseconds]
            foreach text $group {llama::embed $context $text}
            lappend individualSamples [expr {([clock microseconds] - $start) / 1000.0}]
        }
        set batchStats [::embeddingBenchmark::distribution $batchSamples]
        set individualStats [::embeddingBenchmark::distribution $individualSamples]
        set speedup [expr {[dict get $individualStats median_ms] / [dict get $batchStats median_ms]}]
        foreach scenario {native_batch individual_repeated} \
                samples [list $batchSamples $individualSamples] calls [list 1 $count] {
            set stats [::embeddingBenchmark::distribution $samples]
            set median [dict get $stats median_ms]
            set memory [::embeddingBenchmark::memoryStatus]
            set perVector [expr {$median / $count}]
            set vectorsPerSecond [expr {$median > 0 ? 1000.0 * $count / $median : 0.0}]
            set tokensPerSecond [expr {$median > 0 ? 1000.0 * $tokenTotal / $median : 0.0}]
            set first [expr {$scenario eq "native_batch" ? $firstMs : "NA"}]
            puts [join [concat [list $scenario] $common [list $count $tokenTotal $repetitions \
                $modelLoadMs $contextCreateMs $tokenizationMs $first $median \
                [dict get $stats mad_ms] [dict get $stats min_ms] [dict get $stats max_ms] \
                $perVector $vectorsPerSecond $tokensPerSecond $speedup \
                [dict get $memory rss_kb] [dict get $memory peak_rss_kb] $calls]] "\t"]
        }
    }
} finally {
    if {$context ne {}} {llama::context_free $context}
    llama::model_free $model
}
