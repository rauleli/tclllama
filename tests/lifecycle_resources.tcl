#!/usr/bin/env tclsh
# -------------------------------------------------------------------------
# lifecycle_resources.tcl -- Repeated handle lifecycle and Linux memory probe.
#
# Purpose: record process and GGUF mapping metrics at each stage while testing
# repeated model/context creation and destruction. Scenarios are plain, infer,
# snapshot, multisequence, handles, or interpreter.
# Public entry: standalone Tcl script; arguments are packageDir modelPath
# cycles scenario. It prints one key/value row per measurement to stdout.
# Example: tclsh tests/lifecycle_resources.tcl build/local /path/model.gguf 5 infer
# Requirements: Tcl 8.6, tclllama 0.1.0, Linux /proc, and an existing GGUF.
# -------------------------------------------------------------------------

package require Tcl 8.6

# ----------------- Linux process and GGUF mapping measurements ------------

# Read the requested kB-valued metric from a Linux proc status file.
# Returns an integer in KiB, or raises an error if the field is unavailable.
proc lifecycleReadProcKilobytes {path field} {
    set channel [open $path r]
    try {
        while {[gets $channel line] >= 0} {
            if {[regexp [format {^%s:\s+([0-9]+)\s+kB$} $field] $line -> value]} {
                return $value
            }
        }
    } finally {
        close $channel
    }
    error "metric $field is unavailable in $path"
}

# Measure RSS/PSS and model-file mappings for the current process.
# Mapping RSS/PSS are summed from smaps entries whose path contains modelPath.
proc lifecycleMeasure {modelPath} {
    set metrics [dict create \
        rss_kib [lifecycleReadProcKilobytes /proc/self/status VmRSS] \
        hwm_kib [lifecycleReadProcKilobytes /proc/self/status VmHWM]]

    set rollup [open /proc/self/smaps_rollup r]
    try {
        while {[gets $rollup line] >= 0} {
            if {[regexp {^(Pss|Anonymous|Private_Clean|Private_Dirty|Shared_Clean|Shared_Dirty):\s+([0-9]+)\s+kB$} $line -> key value]} {
                dict set metrics [string tolower $key] $value
            }
        }
    } finally {
        close $rollup
    }

    set channel [open /proc/self/smaps r]
    set currentModel 0
    set size 0
    set rss 0
    set pss 0
    set mapCount 0
    set mapSize 0
    set mapRss 0
    set mapPss 0
    try {
        while {[gets $channel line] >= 0} {
            if {[regexp {^[0-9a-f]+-[0-9a-f]+\s} $line]} {
                if {$currentModel} {
                    incr mapCount
                    incr mapSize $size
                    incr mapRss $rss
                    incr mapPss $pss
                }
                set currentModel [expr {[string first $modelPath $line] >= 0}]
                set size 0
                set rss 0
                set pss 0
                continue
            }
            if {$currentModel && [regexp {^(Size|Rss|Pss):\s+([0-9]+)\s+kB$} $line -> field value]} {
                switch -- $field {
                    Size { set size $value }
                    Rss { set rss $value }
                    Pss { set pss $value }
                }
            }
        }
        if {$currentModel} {
            incr mapCount
            incr mapSize $size
            incr mapRss $rss
            incr mapPss $pss
        }
    } finally {
        close $channel
    }

    dict set metrics gguf_maps $mapCount
    dict set metrics gguf_size_kib $mapSize
    dict set metrics gguf_rss_kib $mapRss
    dict set metrics gguf_pss_kib $mapPss
    return $metrics
}

# Print a stable, copyable record for one cycle stage.
proc lifecycleReport {modelPath cycle stage} {
    set metrics [lifecycleMeasure $modelPath]
    set fields {rss_kib hwm_kib pss anonymous private_clean private_dirty shared_clean shared_dirty}
    set fields [lmap field $fields {
        if {[dict exists $metrics $field]} { dict get $metrics $field } else { "n/a" }
    }]
    puts "cycle=$cycle stage=$stage rss_kib=[lindex $fields 0] hwm_kib=[lindex $fields 1] pss_kib=[lindex $fields 2] anonymous_kib=[lindex $fields 3] private_clean_kib=[lindex $fields 4] private_dirty_kib=[lindex $fields 5] shared_clean_kib=[lindex $fields 6] shared_dirty_kib=[lindex $fields 7] gguf_maps=[dict get $metrics gguf_maps] gguf_size_kib=[dict get $metrics gguf_size_kib] gguf_rss_kib=[dict get $metrics gguf_rss_kib] gguf_pss_kib=[dict get $metrics gguf_pss_kib]"
    return $metrics
}

# Create a small CPU context; multisequence enables independent fork checks.
proc lifecycleCreateHandle {modelPath scenario} {
    if {$scenario eq "multisequence"} {
        return [llama::init $modelPath 512 -options [dict create \
            n_seq_max 2 sequence_sampling true n_batch 32 \
            n_threads 2 n_threads_batch 2]]
    }
    return [llama::init $modelPath 512 -options [dict create \
        n_batch 32 n_threads 2 n_threads_batch 2]]
}

# Execute the selected workload, always freeing any live handles afterward.
proc lifecycleRunScenario {modelPath scenario packageDir cycle} {
    switch -- $scenario {
        plain - infer - snapshot - multisequence {
            set handle [lifecycleCreateHandle $modelPath $scenario]
            try {
                lifecycleReport $modelPath $cycle after_init
                if {$scenario eq "infer"} {
                    llama::generate $handle "lifecycle check" -max_tokens 1
                } elseif {$scenario eq "snapshot"} {
                    llama::generate $handle "lifecycle snapshot" -max_tokens 1
                    llama::snapshot_save $handle
                    llama::snapshot_discard $handle
                } elseif {$scenario eq "multisequence"} {
                    set tokens [llama::tokenize $handle "lifecycle branches"]
                    llama::sequence $handle prefill 0 $tokens
                    llama::sequence $handle generate 0 -options {
                        temperature 0 seed 101
                    } -max_tokens 1
                    llama::sequence $handle fork 0 1
                    llama::sequence $handle generate 0 -max_tokens 1
                    llama::sequence $handle generate 1 -max_tokens 1
                }
                lifecycleReport $modelPath $cycle after_workload
            } finally {
                llama::free $handle
            }
            set afterFree [lifecycleReport $modelPath $cycle after_free]
            if {[dict get $afterFree gguf_maps] != 0} {
                error "GGUF mappings remain after freeing the only handle"
            }
        }
        handles {
            set first [lifecycleCreateHandle $modelPath plain]
            set second {}
            try {
                lifecycleReport $modelPath $cycle after_first_handle
                set second [lifecycleCreateHandle $modelPath plain]
                lifecycleReport $modelPath $cycle after_second_handle
                llama::free $first
                set first {}
                set afterFirstFree [lifecycleReport $modelPath $cycle after_first_free]
                if {[dict get $afterFirstFree gguf_maps] == 0} {
                    error "second handle lost its GGUF mapping after first handle free"
                }
                llama::generate $second "second handle remains live" -max_tokens 1
            } finally {
                if {$first ne {}} { llama::free $first }
                if {$second ne {}} { llama::free $second }
            }
            set afterBothFree [lifecycleReport $modelPath $cycle after_both_free]
            if {[dict get $afterBothFree gguf_maps] != 0} {
                error "GGUF mappings remain after freeing both handles"
            }
        }
        interpreter {
            set child [interp create]
            try {
                interp eval $child [list lappend auto_path $packageDir]
                interp eval $child {package require tclllama 0.1.0}
                interp eval $child [list set modelPath $modelPath]
                interp eval $child {
                    set handle [llama::init $modelPath 512]
                    llama::generate $handle "interpreter teardown" -max_tokens 1
                }
                lifecycleReport $modelPath $cycle before_interp_delete
                interp delete $child
                lifecycleReport $modelPath $cycle after_interp_delete
            } finally {
                if {[interp exists $child]} { interp delete $child }
            }
            set afterDelete [lifecycleMeasure $modelPath]
            if {[dict get $afterDelete gguf_maps] != 0} {
                error "GGUF mappings remain after deleting the owning interpreter"
            }
        }
        default {
            error "scenario must be plain, infer, snapshot, multisequence, handles, or interpreter"
        }
    }
}

# ----------------- Script entry point --------------------------------------

if {$argc < 2 || $argc > 4} {
    puts stderr "Usage: tclsh lifecycle_resources.tcl packageDir modelPath ?cycles? ?scenario?"
    puts stderr "Scenarios: plain, infer, snapshot, multisequence, handles, interpreter"
    exit 2
}

set packageDir [file normalize [lindex $argv 0]]
set modelPath [file normalize [lindex $argv 1]]
set cycles [expr {$argc >= 3 ? [lindex $argv 2] : 5}]
set scenario [expr {$argc >= 4 ? [lindex $argv 3] : "infer"}]
if {![string is integer -strict $cycles] || $cycles < 1 || $cycles > 20} {
    error "cycles must be an integer from 1 through 20"
}
if {![file isfile $modelPath]} {
    error "model file does not exist: $modelPath"
}
if {![file isfile /proc/self/smaps_rollup] || ![file isfile /proc/self/smaps]} {
    error "this probe requires Linux /proc/self/smaps and smaps_rollup"
}
lappend auto_path $packageDir
package require tclllama 0.1.0

puts "model=$modelPath scenario=$scenario cycles=$cycles package=$packageDir"
for {set cycle 1} {$cycle <= $cycles} {incr cycle} {
    lifecycleReport $modelPath $cycle before_init
    lifecycleRunScenario $modelPath $scenario $packageDir $cycle
}
