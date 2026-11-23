#!/usr/bin/env tclsh
# -----------------------------------------------------------------------------
# inspect.tcl -- Portable host and tclllama capability report
#
# Purpose: report OS, CPU instruction flags, RAM, compiler, Tcl, linked package
#          libraries, and optional GPU/backend discovery without assuming that
#          any diagnostic utility is installed.
# Public interface: tclsh inspect.tcl ?-package-dir path? ?-output path?
# Requirements: Tcl 8.6. Writes TSV; unavailable probes are recorded, not fatal.
# -----------------------------------------------------------------------------

namespace eval ::labInspect {
    # Emit one escaped TSV record. Tabs/newlines in values are replaced so each
    # probe remains a single independently consumable row.
    proc row {channel key state value reason} {
        set clean [string map [list "\t" { } "\r" { } "\n" { }] $value]
        set why [string map [list "\t" { } "\r" { } "\n" { }] $reason]
        puts $channel "$key\t$state\t$clean\t$why"
    }

    # Run an optional command with a short timeout. Missing tools and command
    # failures become report rows rather than terminating the inspection.
    proc probeCommand {channel key command arguments} {
        if {[auto_execok $command] eq {}} {
            row $channel $key unsupported {} "$command is unavailable"
            return
        }
        if {[catch {exec -keepnewline $command {*}$arguments 2>@1} output]} {
            row $channel $key unsupported $output "$command returned an error"
            return
        }
        row $channel $key pass [string trim $output] {}
    }

    # Read a Linux proc/sys text file when available, otherwise record that the
    # host does not expose that source.
    proc readFile {path} {
        if {![file readable $path]} {
            return {}
        }
        set input [open $path r]
        try {
            set contents [read $input]
        } finally {
            close $input
        }
        return $contents
    }

    # Record package information without initializing a model or context.
    proc probePackage {channel packageDir} {
        if {$packageDir eq {}} {return}
        if {![file isdirectory $packageDir]} {
            row $channel tclllama unsupported {} "package directory not found: $packageDir"
            return
        }
        lappend ::auto_path [file normalize $packageDir]
        if {[catch {package require tclllama 0.1.0} message]} {
            row $channel tclllama fail $message {package load failed}
            return
        }
        row $channel tclllama pass [package provide tclllama] {}
        foreach {key command} {
            engine_info {llama::engine_info}
            backend_info {llama::backend_info}
            devices {llama::devices}
        } {
            if {[catch {{*}$command} value]} {
                row $channel "llama_$key" unsupported $value {native query failed}
            } else {
                row $channel "llama_$key" pass $value {}
            }
        }
    }
}

set packageDir {}
set outputPath {}
if {$argc % 2} {
    puts stderr "Usage: inspect.tcl ?-package-dir path? ?-output path?"
    exit 2
}
foreach {option value} $argv {
    switch -- $option {
        -package-dir {set packageDir $value}
        -output {set outputPath $value}
        default {
            puts stderr "Unknown option: $option"
            exit 2
        }
    }
}

if {$outputPath eq {}} {
    set channel stdout
} else {
    file mkdir [file dirname [file normalize $outputPath]]
    set channel [open $outputPath w]
}
try {
    puts $channel "key\tstatus\tvalue\treason"
    ::labInspect::row $channel hostname pass [info hostname] {}
    ::labInspect::row $channel os pass $::tcl_platform(os) {}
    ::labInspect::row $channel os_version pass $::tcl_platform(osVersion) {}
    ::labInspect::row $channel machine pass $::tcl_platform(machine) {}
    ::labInspect::row $channel tcl_version pass [info patchlevel] {}
    set release [::labInspect::readFile /etc/os-release]
    if {$release eq {}} {
        ::labInspect::row $channel os_release unsupported {} {/etc/os-release unavailable}
    } else {
        set releaseName [expr {[regexp -line {^(PRETTY_NAME|NAME)=?(.*)$} $release -> key value] ? $value : $release}]
        ::labInspect::row $channel os_release pass $releaseName {}
    }
    set cpuInfo [::labInspect::readFile /proc/cpuinfo]
    set memInfo [::labInspect::readFile /proc/meminfo]
    set swapInfo [::labInspect::readFile /proc/swaps]

    set flags {}
    if {[regexp -line {^flags\s*:\s*(.*)$} $cpuInfo -> flags]} {
        set flags [lsort -unique [split $flags]]
        foreach {feature aliases} {
            sse {sse}
            sse2 {sse2}
            sse3 {sse3 pni}
            ssse3 {ssse3}
            sse4_1 {sse4_1}
            sse4_2 {sse4_2}
            sse4a {sse4a}
            avx {avx}
            avx2 {avx2}
            avx512f {avx512f}
            avx512bw {avx512bw}
            avx512dq {avx512dq}
            avx512vl {avx512vl}
            avx512_vnni {avx512_vnni}
        } {
            set present 0
            foreach alias $aliases {
                if {[lsearch -exact $flags $alias] >= 0} {set present 1}
            }
            ::labInspect::row $channel "cpu_$feature" \
                [expr {$present ? "pass" : "unsupported"}] \
                [expr {$present ? "present" : "absent"}] {reported CPU flags}
        }
    } else {
        ::labInspect::row $channel cpu_flags unsupported {} {CPU flags not available in /proc/cpuinfo}
    }

    if {[regexp -line {^model name\s*:\s*(.*)$} $cpuInfo -> value]} {
        ::labInspect::row $channel cpu_model pass $value {}
    } elseif {[regexp -line {^Hardware\s*:\s*(.*)$} $cpuInfo -> value]} {
        ::labInspect::row $channel cpu_model pass $value {}
    } else {
        ::labInspect::row $channel cpu_model unsupported {} {CPU model string unavailable}
    }

    set ram {}
    if {[regexp -line {^MemTotal:\s*(.*)$} $memInfo -> ram]} {
        ::labInspect::row $channel ram pass [string trim $ram] {}
    } else {
        ::labInspect::row $channel ram unsupported {} {MemTotal not available}
    }
    foreach field {SwapTotal SwapFree} {
        if {[regexp -line "^$field:\\s*(.*)$" $memInfo -> value]} {
            ::labInspect::row $channel [string tolower $field] pass [string trim $value] {}
        } else {
            ::labInspect::row $channel [string tolower $field] unsupported {} {not reported by /proc/meminfo}
        }
    }
    if {[auto_execok g++] ne {}} {
        ::labInspect::probeCommand $channel compiler [auto_execok g++] {--version}
    } elseif {[auto_execok c++] ne {}} {
        ::labInspect::probeCommand $channel compiler [auto_execok c++] {--version}
    } else {
        ::labInspect::row $channel compiler unsupported {} {no C++ compiler found}
    }
    foreach {key command arguments} {
        lscpu lscpu {-J}
        nvidia_smi nvidia-smi {--query-gpu=name,driver_version --format=csv,noheader}
        rocm_smi rocm-smi {--showproductname}
        opencl clinfo {-l}
    } {
        ::labInspect::probeCommand $channel $key $command $arguments
    }

    if {$packageDir ne {}} {
        ::labInspect::probePackage $channel $packageDir
        set extension [file join $packageDir tclllama.so]
        if {[file isfile $extension]} {
            ::labInspect::probeCommand $channel linked_libraries ldd [list $extension]
        } else {
            ::labInspect::row $channel linked_libraries unsupported {} "$extension not found"
        }
    }
} finally {
    if {$channel ne "stdout"} {close $channel}
}
