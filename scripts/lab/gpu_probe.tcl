#!/usr/bin/env tclsh
# gpu_probe.tcl -- Bounded child-process GPU inference attempt for lab runner.
# Usage: gpu_probe.tcl PACKAGE_DIR MODEL_PATH
# Requires Tcl 8.6 and a tclllama 0.1.0 build with a registered GPU backend.
# The caller captures this process's native logs and verifies the model hash.
# A successful inference does not itself certify that GPU compute was used.

if {$argc != 2} {puts stderr {usage: gpu_probe.tcl PACKAGE_DIR MODEL_PATH}; exit 2}
lassign $argv packageDir modelPath
lappend auto_path [file normalize $packageDir]
if {[catch {package require tclllama 0.1.0} message]} {
    puts stderr "GPU_PROBE package_error=$message"
    exit 2
}
set loadedModule {}
foreach item [info loaded] {
    lassign $item path init
    if {[string equal -nocase $init Tclllama]} {set loadedModule [file normalize $path]}
}
if {$loadedModule eq {} || [file dirname $loadedModule] ne [file normalize $packageDir]} {
    puts stderr "GPU_PROBE module_error=loaded_from_unexpected_directory"
    exit 2
}
puts "GPU_PROBE loaded=$loadedModule"
puts "GPU_PROBE backends=[llama::backend_info]"
set selected {}
foreach device [llama::devices] {
    puts "GPU_PROBE device=$device"
    if {$selected eq {} && [dict get $device type] in {gpu igpu}} {set selected $device}
}
if {$selected eq {}} {puts {GPU_PROBE status=no_registered_gpu}; exit 3}
puts "GPU_PROBE selected=$selected model=[file normalize $modelPath]"
set handle {}
set failure 0
try {
    set options [list devices [list [dict get $selected index]] n_gpu_layers -1 n_threads 2 n_threads_batch 2]
    set handle [llama::init $modelPath 512 -options $options]
    set reply [llama::generate $handle Hello -max_tokens 4]
    if {$reply eq {}} {error {inference produced no text}}
    puts {GPU_PROBE status=inference_completed}
} on error {message} {
    puts stderr "GPU_PROBE inference_error=$message"
    set failure 1
} finally {
    if {$handle ne {}} {llama::free $handle}
}
exit $failure
