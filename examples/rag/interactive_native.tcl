#!/usr/bin/env tclsh
# interactive_native.tcl -- Interactive RAG using native llama::chat.
# Uses private arguments documented in lib/app.tcl; append --verbose or --debug
# to expose native llama.cpp diagnostics. Default output keeps warnings/errors.
package require Tcl 8.6
package require tclllama 0.1.0
set log_level warn
if {[llength $argv] && [lindex $argv end] in {--verbose --debug}} {
    set option [lindex $argv end]
    set argv [lrange $argv 0 end-1]
    if {$option eq "--verbose"} {set log_level info} else {set log_level debug}
}
llama::log_level $log_level
set here [file dirname [file normalize [info script]]]
if {![llength $argv] || [lindex $argv 0] ni {sqlite mysql}} {error "first argument must be sqlite or mysql"}
source [file join $here lib [lindex $argv 0]_store.tcl]
foreach name {binding retrieve generate repl app} {source [file join $here lib $name.tcl]}
::rag::app::run native {*}$argv
