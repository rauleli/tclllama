#!/usr/bin/env tclsh
# -----------------------------------------------------------------------------
# embeddings_characterize.tcl — Numerical batch/individual comparison
#
# Purpose: compare native batches against individual embedding calls using a
#          deterministic multilingual corpus, without imposing semantic scores.
# Entry point: command-line Tcl script that writes summary TSV to stdout.
# Requirements: Tcl 8.6, tclllama 0.1.0, and one local encoder GGUF.
# Usage: tclsh tests/embeddings_characterize.tcl build/local model.gguf label
#        ?taskPrefix?
# -----------------------------------------------------------------------------

if {$argc < 3 || $argc > 4} {
    puts stderr "Usage: embeddings_characterize.tcl packageDir modelPath label ?taskPrefix?"
    exit 2
}
set packageDir [file normalize [lindex $argv 0]]
set modelPath [file normalize [lindex $argv 1]]
set modelLabel [lindex $argv 2]
set taskPrefix [expr {$argc == 4 ? [lindex $argv 3] : {}}]
if {![file isfile $modelPath]} {
    puts stderr "model file is not readable: $modelPath"
    exit 2
}

lappend auto_path $packageDir
package require tclllama 0.1.0

namespace eval ::embeddingCharacterize {}

# Compare dimensions numerically; relative error excludes near-zero reference
# coordinates to avoid dividing by values dominated by float noise.
proc ::embeddingCharacterize::compare {left right} {
    if {[llength $left] != [llength $right]} {
        return [dict create valid 0 max_abs NA max_relative NA cosine NA]
    }
    set maxAbsolute 0.0
    set maxRelative 0.0
    set hasRelative 0
    set dot 0.0
    set leftNorm 0.0
    set rightNorm 0.0
    foreach a $left b $right {
        set a [expr {double($a)}]
        set b [expr {double($b)}]
        if {![expr {abs($a) < Inf && abs($b) < Inf}]} {
            return [dict create valid 0 max_abs NA max_relative NA cosine NA]
        }
        set difference [expr {abs($a - $b)}]
        if {$difference > $maxAbsolute} {set maxAbsolute $difference}
        if {abs($b) > 1.0e-8} {
            set relative [expr {$difference / abs($b)}]
            if {$relative > $maxRelative} {set maxRelative $relative}
            set hasRelative 1
        }
        set dot [expr {$dot + $a * $b}]
        set leftNorm [expr {$leftNorm + $a * $a}]
        set rightNorm [expr {$rightNorm + $b * $b}]
    }
    if {$leftNorm <= 0.0 || $rightNorm <= 0.0} {
        set cosine NA
    } else {
        set cosine [expr {$dot / sqrt($leftNorm * $rightNorm)}]
    }
    return [dict create valid 1 max_abs $maxAbsolute \
        max_relative [expr {$hasRelative ? $maxRelative : "NA"}] cosine $cosine]
}

# Calculate a Euclidean norm for characterization only; no vector is modified.
proc ::embeddingCharacterize::norm {vector} {
    set squared 0.0
    foreach value $vector {set squared [expr {$squared + double($value) * double($value)}]}
    return [expr {sqrt($squared)}]
}

set texts [list \
    {A short English sentence about language models.} \
    {Texto breve en español: acción, pingüino, café y árbol.} \
    {A medium document describes native inference, sequence pooling, and vector representations.} \
    {La inferencia por lotes conserva el orden y separa las secuencias del contexto.} \
    {A repeated document about language models and native embeddings.} \
    {Texto similar en español: modelos nativos, vectores y secuencias independientes.} \
    {A longer entry combines UTF-8 symbols: naïve, café, and 東京 with ordinary English words.} \
    {Documento distinto de longitud media para comparar resultados numéricos y posiciones.}]
if {$taskPrefix ne {}} {
    set prefixed {}
    foreach text $texts {lappend prefixed "$taskPrefix$text"}
    set texts $prefixed
}

set model [llama::model_load $modelPath]
set context {}
try {
    set context [llama::embedding_context_create $model -options {
        pooling_type model attention_type model n_threads 2 n_threads_batch 2
        n_batch 2048 n_ubatch 512 n_seq_max 8
    }]
    set info [llama::info $context]
    set dimension [dict get $info model n_embd_out]
    set batched [llama::embed_batch $context $texts]
    set individual {}
    set tokenCounts {}
    foreach text $texts {
        lappend individual [llama::embed $context $text]
        lappend tokenCounts [llength [llama::tokenize $context $text]]
    }
    puts "model\tdimension\ttexts\ttokens\tbatch_native_calls\tindividual_native_calls\tmax_abs_error\tmax_relative_error_gt_1e-8\tminimum_cosine\tmin_l2_norm\tmax_l2_norm"
    foreach count {2 4 8} {
        set totalTokens 0
        set maxAbsolute 0.0
        set maxRelative 0.0
        set hasRelative 0
        set minimumCosine 1.0
        set minNorm Inf
        set maxNorm 0.0
        set valid 1
        for {set i 0} {$i < $count} {incr i} {
            incr totalTokens [lindex $tokenCounts $i]
            set comparison [::embeddingCharacterize::compare \
                [lindex $batched $i] [lindex $individual $i]]
            if {![dict get $comparison valid] ||
                [llength [lindex $batched $i]] != $dimension} {set valid 0}
            if {[dict get $comparison max_abs] ne "NA" &&
                [dict get $comparison max_abs] > $maxAbsolute} {
                set maxAbsolute [dict get $comparison max_abs]
            }
            if {[dict get $comparison max_relative] ne "NA"} {
                if {[dict get $comparison max_relative] > $maxRelative} {
                    set maxRelative [dict get $comparison max_relative]
                }
                set hasRelative 1
            }
            if {[dict get $comparison cosine] ne "NA" &&
                [dict get $comparison cosine] < $minimumCosine} {
                set minimumCosine [dict get $comparison cosine]
            }
            set currentNorm [::embeddingCharacterize::norm [lindex $batched $i]]
            if {$currentNorm < $minNorm} {set minNorm $currentNorm}
            if {$currentNorm > $maxNorm} {set maxNorm $currentNorm}
        }
        if {!$valid} {error "invalid vector dimensions or non-finite output in first $count texts"}
        set relative [expr {$hasRelative ? $maxRelative : "NA"}]
        puts [join [list $modelLabel $dimension $count $totalTokens 1 $count \
            $maxAbsolute $relative $minimumCosine $minNorm $maxNorm] "\t"]
    }
} finally {
    if {$context ne {}} {llama::context_free $context}
    llama::model_free $model
}
