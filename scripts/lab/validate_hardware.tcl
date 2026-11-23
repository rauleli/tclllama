#!/usr/bin/env tclsh
# validate_hardware.tcl -- Bounded hardware validation with existing lab suites.
# CPU: -profile cpu-baseline|cpu-avx2|cpu-avx512 -llama-build PATH
#      -build-log PATH -output NEW_DIR ?-model-store PATH?
# GPU: -profile gpu -package-dir PATH -output NEW_DIR ?-model-store PATH?
# Requires Tcl 8.6, Tcllib json, timeout, sha256sum, mktemp, GNU ar, make and compilers.
# CPU builds are verified from verbose logs; GPU offload remains unverified.

namespace eval ::hardwareValidation {
    # Write a TSV row, rejecting embedded delimiters in all fields.
    proc row {channel args} {
        set fields {}
        foreach value $args {
            lappend fields [string map [list "\t" { } "\n" { } "\r" { }] $value]
        }
        puts $channel [join $fields "\t"]
        flush $channel
    }

    # Read a text file or raise an error that the caller reports as a failure.
    proc readFile {path} {
        set channel [open $path r]
        try {return [read $channel]} finally {close $channel}
    }

    # Remove make dependency-file arguments that are present in verbose
    # invocations but not in CMake's compilation database. Inputs are shell
    # words from CMake-generated commands; unsupported quoting fails closed.
    proc compilerWords {command} {
        if {[regexp {["'\\]} $command]} {error {unsupported quoting in compiler invocation}}
        set result {}
        set words [regexp -all -inline {\S+} $command]
        for {set i 0} {$i < [llength $words]} {incr i} {
            set word [lindex $words $i]
            if {$word in {-MT -MF}} {incr i; continue}
            if {$word eq "-MD"} {continue}
            lappend result $word
        }
        return $result
    }

    # Every compiled object assigned to an archive must appear in it. Checking
    # members only would allow an incomplete archive to pass certification.
    proc requireArchiveCoverage {objects matched target archive} {
        dict for {object unused} $objects {
            if {[string first $target $object] == 0 && ![dict exists $matched $object]} {
                error "compiled object missing from archive $archive: $object"
            }
        }
    }

    # Certify only archives whose members match objects with corresponding
    # verbose compiler invocations. An incomplete log or ambiguous object name
    # raises an error; the caller must not report profile_config as pass.
    proc verifyEngine {engine profile buildLog} {
        package require json
        set database [::json::json2dict [readFile [file join $engine compile_commands.json]]]
        set log [readFile $buildLog]
        set objects {}
        foreach item $database {
            if {![dict exists $item output]} {error {compilation database lacks output}}
            set object [dict get $item output]
            set expected [compilerWords [dict get $item command]]
            set directory [dict get $item directory]
            set outputIndex [lsearch -exact $expected -o]
            set sourceIndex [lsearch -exact $expected -c]
            if {$outputIndex < 0 || $sourceIndex < 0 || $outputIndex + 1 >= [llength $expected] || $sourceIndex + 1 >= [llength $expected]} {
                error "compiler input or output is missing: $object"
            }
            set declaredOutput [file normalize [file join $engine $object]]
            set commandOutput [file normalize [file join $directory [lindex $expected [expr {$outputIndex + 1}]]]]
            set commandSource [file normalize [file join $directory [lindex $expected [expr {$sourceIndex + 1}]]]]
            if {$declaredOutput ne $commandOutput || $commandSource ne [file normalize [dict get $item file]] || [string first "$engine/" $declaredOutput] != 0} {
                error "compiler input/output does not match database: $object"
            }
            if {[lsearch -exact $expected -march=x86-64] < 0 || [lsearch -exact $expected -mtune=generic] < 0 || [lsearch -exact $expected -fPIC] < 0} {
                error "missing baseline flags: $object"
            }
            foreach word $expected {
                if {[string match -m* $word]} {
                    set allowed {-march=x86-64 -mtune=generic}
                    if {$profile in {cpu-avx2 cpu-avx512} && [string match *ggml-cpu.dir/* $object]} {
                        lappend allowed -mavx -mavx2 -mfma -mf16c -mbmi2
                        if {$profile eq "cpu-avx512"} {lappend allowed -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw}
                    }
                    if {$word ni $allowed} {error "disallowed ISA flag $word in $object"}
                }
            }
            if {$profile in {cpu-avx2 cpu-avx512} && [string match *ggml-cpu.dir/* $object]} {
                set requiredFlags {-mavx -mavx2 -mfma -mf16c -mbmi2}
                if {$profile eq "cpu-avx512"} {lappend requiredFlags -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw}
                foreach flag $requiredFlags {
                    if {[lsearch -exact $expected $flag] < 0} {error "missing $flag in $object"}
                }
            }
            set prefix "cd $directory && "
            set match 0
            foreach line [split $log "\n"] {
                if {![string match "${prefix}*" $line]} {continue}
                if {[catch {compilerWords [string range $line [string length $prefix] end]} actual]} {continue}
                if {$actual eq $expected} {incr match}
            }
            if {$match != 1} {error "expected one matching executed compiler invocation for $object; found $match"}
            if {[dict exists $objects $object]} {error "duplicate compilation database output: $object"}
            dict set objects $object 1
        }
        set total 0
        foreach {archive target} {
            src/libllama.a src/CMakeFiles/llama.dir/
            ggml/src/libggml.a ggml/src/CMakeFiles/ggml.dir/
            ggml/src/libggml-cpu.a ggml/src/CMakeFiles/ggml-cpu.dir/
            ggml/src/libggml-base.a ggml/src/CMakeFiles/ggml-base.dir/
        } {
            set path [file join $engine $archive]
            set targetName [file tail [string trimright $target /]]
            set linkScript [file join $engine [file dirname $archive] CMakeFiles $targetName link.txt]
            set archiveCommand [lindex [split [readFile $linkScript] "\n"] 0]
            if {$archiveCommand eq {} || [string first $archiveCommand $log] < 0} {error "archiver invocation not found in verbose log: $archive"}
            set members [split [string trim [exec ar t $path]] "\n"]
            if {![llength $members] || [lindex $members 0] eq {}} {error "empty archive: $archive"}
            set occurrences {}
            set matchedObjects {}
            foreach member $members {
                dict incr occurrences $member
                set instance [dict get $occurrences $member]
                set matches {}
                dict for {object unused} $objects {
                    if {[string first $target $object] == 0 && [file tail $object] eq $member} {lappend matches $object}
                }
                # GNU ar permits repeated member names; extract the exact Nth
                # instance, then match it to one unused object by content.
                set tempPath [string trim [exec mktemp -d]]
                set oldDir [pwd]
                try {
                    cd $tempPath
                    exec ar xN $instance $path $member
                    set actual [lindex [exec sha256sum -- [file join $tempPath $member]] 0]
                } finally {
                    cd $oldDir
                    file delete -force $tempPath
                }
                set exact {}
                foreach object $matches {
                    if {[dict exists $matchedObjects $object]} {continue}
                    set expected [lindex [exec sha256sum -- [file join $engine $object]] 0]
                    if {$actual eq $expected} {lappend exact $object}
                }
                if {[llength $exact] != 1} {error "archive member has no unique unused matching object: $archive:$member instance $instance"}
                dict set matchedObjects [lindex $exact 0] 1
                incr total
            }
            requireArchiveCoverage $objects $matchedObjects $target $archive
        }
        return "[llength $database] executed compile commands and $total archive members verified"
    }

    # XCR0 bits 1, 2, 5, 6 and 7 are needed for AVX-512 state. Invalid or
    # unavailable helper output is unknown, never evidence of support.
    proc xcr0Decision {probe {requiredMask 0xe6}} {
        if {![regexp {^xsave=(0|1) osxsave=(0|1) xcr0=(unknown|0x[0-9a-fA-F]+)$} [string trim $probe] -> xsave osxsave xcr0]} {
            return unknown
        }
        if {!$xsave || !$osxsave} {return unsupported}
        if {$xcr0 eq "unknown"} {return unknown}
        if {($xcr0 & $requiredMask) != $requiredMask} {return unsupported}
        return verified
    }

    # Classify AVX-512 for the pinned ggml profile. Linux flags are inspected
    # for every visible CPU; they do not independently expose XCR0 state.
    proc avx512Decision {cpuInfo osState} {
        set required {avx512f avx512cd avx512vl avx512dq avx512bw}
        set cpus 0
        foreach line [split $cpuInfo "\n"] {
            if {![regexp {^flags\s*:\s*(.*)$} $line -> flags]} {continue}
            incr cpus
            foreach flag $required {
                if {[lsearch -exact [split $flags] $flag] < 0} {
                    return [list unsupported "visible CPU lacks $flag"]
                }
            }
        }
        if {!$cpus} {return [list not_run {CPU flags unavailable}]}
        set processors [regexp -all -line {^processor[ \t]*:} $cpuInfo]
        if {$processors > 0 && $processors != $cpus} {
            return [list not_run {CPU flags missing for one or more visible processors}]
        }
        if {$osState eq "unsupported"} {return [list unsupported {OS does not enable required extended register state}]}
        if {$osState ne "verified"} {return [list not_run {OS extended register state not independently verified}]}
        return [list pass {required CPU flags and OS state verified}]
    }

    # Inspect PCI display-class devices. A display controller is identification
    # evidence only: it does not establish a usable compute backend or offload.
    proc gpuInventory {root} {
        if {![file isdirectory $root] || ![file readable $root]} {
            return [list not_run {PCI inventory unavailable}]
        }
        set found {}
        foreach device [glob -nocomplain -directory $root *] {
            if {[catch {set class [string trim [readFile [file join $device class]]]}]} {continue}
            if {![string match 0x03* $class]} {continue}
            set vendor unknown
            catch {set vendor [string trim [readFile [file join $device vendor]]]}
            set driver unbound
            if {[file exists [file join $device driver]]} {
                catch {set driver [file tail [file readlink [file join $device driver]]]}
            }
            lappend found "[file tail $device]:class=$class,vendor=$vendor,driver=$driver"
        }
        if {![llength $found]} {return [list not_run {no PCI display controller observed; other devices may be hidden}]}
        return [list pass [join $found {; }]]
    }

    # A registered GPU, successful inference and the upstream layer message
    # remain insufficient: the pinned engine derives that message from the
    # requested layer count. Keep offload unverified until same-process backend
    # execution evidence is available.
    proc gpuOffloadDecision {gpuRegistered inferenceSucceeded nativeLog} {
        if {!$gpuRegistered || !$inferenceSucceeded} {
            return [list not_run {GPU backend or successful inference not established}]
        }
        if {![regexp {offloaded ([0-9]+)/([0-9]+) layers to GPU} $nativeLog -> offloaded total] || $offloaded < 1 || $total < $offloaded} {
            return [list not_run {positive native GPU layer offload evidence missing}]
        }
        return [list not_run "native log reports $offloaded/$total requested layers; GPU execution is not independently proven"]
    }
}

set repo [file normalize [file join [file dirname [info script]] ../..]]
set opts [dict create -profile {} -llama-build {} -build-log {} -package-dir {} -output {} -model-store [file join $repo ../datos/modelos/blobs]]
if {$argc % 2} {puts stderr "Expected option/value pairs"; exit 2}
foreach {key value} $argv {
    if {![dict exists $opts $key]} {puts stderr "Unknown option: $key"; exit 2}
    dict set opts $key $value
}
set profile [dict get $opts -profile]
if {$profile ni {cpu-baseline cpu-avx2 cpu-avx512 gpu} || [dict get $opts -output] eq {} || ($profile eq "gpu" && [dict get $opts -package-dir] eq {}) || ($profile ne "gpu" && [dict get $opts -llama-build] eq {})} {
    puts stderr "Usage: validate_hardware.tcl -profile cpu-baseline|cpu-avx2|cpu-avx512 -llama-build PATH -build-log PATH -output NEW_DIR; GPU: -profile gpu -package-dir PATH -output NEW_DIR"
    exit 2
}
set out [file normalize [dict get $opts -output]]
if {[file exists $out]} {puts stderr "Output path already exists: $out"; exit 2}
file mkdir $out
set result [open [file join $out summary.tsv] w]
puts $result "check\ttarget\tstatus\tdetail"
set engine [expr {$profile eq "gpu" ? {} : [file normalize [dict get $opts -llama-build]]}]
set source [file normalize [file join $repo ../dependencias/llama.cpp]]
set pinned 1719dbb0ee7bdaf56de083079d3613bb670160a5
set success 1
try {
    set inspect [file join $repo scripts lab inspect.tcl]
    if {[catch {exec [info nameofexecutable] $inspect -output [file join $out environment.tsv]} message]} {error "inspection failed: $message"}
    set cpu [::hardwareValidation::readFile /proc/cpuinfo]
    set xcrState unknown
    set avxState unknown
    set helper [file join $out x86_xcr0]
    set helperSource [file join $repo scripts lab x86_xcr0.c]
    if {$::tcl_platform(machine) eq "x86_64" && [auto_execok cc] ne {}} {
        if {![catch {exec timeout 30s cc -O2 -march=x86-64 -mtune=generic $helperSource -o $helper 2>@1} compileOutput] &&
            ![catch {exec timeout 5s $helper 2>@1} probe]} {
            set xcrState [::hardwareValidation::xcr0Decision $probe]
            set avxState [::hardwareValidation::xcr0Decision $probe 0x6]
            ::hardwareValidation::row $result cpu xcr0 [expr {$xcrState eq "unknown" ? "not_run" : "pass"}] $probe
        } else {
            ::hardwareValidation::row $result cpu xcr0 not_run {helper compile or execution failed}
        }
    } else {
        ::hardwareValidation::row $result cpu xcr0 not_run {x86_64 or C compiler unavailable}
    }
    lassign [::hardwareValidation::avx512Decision $cpu $xcrState] avxStatus avxDetail
    ::hardwareValidation::row $result cpu-avx512 eligibility $avxStatus $avxDetail
    lassign [::hardwareValidation::gpuInventory /sys/bus/pci/devices] gpuStatus gpuDetail
    ::hardwareValidation::row $result gpu pci_device $gpuStatus $gpuDetail
    ::hardwareValidation::row $result gpu backend not_run {backend enumeration not performed}
    ::hardwareValidation::row $result gpu offload not_run {no native offload proof performed}
    if {$profile eq "gpu"} {
        set hash 74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45
        set model [file normalize [file join [dict get $opts -model-store] sha256-$hash]]
        if {![file isfile $model] || [lindex [exec sha256sum -- $model] 0] ne $hash} {error {Llama model missing or SHA-256 mismatch}}
        set gpuLog [file join $out gpu-probe.log]
        set probeScript [file join $repo scripts lab gpu_probe.tcl]
        set packageDir [file normalize [dict get $opts -package-dir]]
        set rc [catch {exec timeout --signal=TERM --kill-after=5s 300s env -u TCLLIBPATH [info nameofexecutable] $probeScript $packageDir $model > $gpuLog 2>@1} message]
        set probeOutput [::hardwareValidation::readFile $gpuLog]
        if {[regexp -line {^GPU_PROBE loaded=(.+)$} $probeOutput -> gpuModule] && [file isfile $gpuModule] && [file dirname [file normalize $gpuModule]] eq $packageDir} {
            ::hardwareValidation::row $result gpu loaded_module pass "path=$gpuModule; sha256=[lindex [exec sha256sum -- $gpuModule] 0]"
        } else {
            ::hardwareValidation::row $result gpu loaded_module not_run {module identity unavailable in child output}
        }
        if {[string first {GPU_PROBE status=no_registered_gpu} $probeOutput] >= 0} {
            ::hardwareValidation::row $result gpu registered_backend not_run {no registered GPU in supplied package}
        } elseif {$rc} {
            ::hardwareValidation::row $result gpu inference fail "probe process failed: $message; log=gpu-probe.log"
        } elseif {[regexp -line {^GPU_PROBE status=inference_completed$} $probeOutput]} {
            ::hardwareValidation::row $result gpu inference pass "child completed; model sha256=$hash; log=gpu-probe.log"
        } else {
            ::hardwareValidation::row $result gpu inference fail {probe returned zero without completion marker}
        }
        ::hardwareValidation::row $result gpu certification not_run {native execution evidence unavailable in pinned public API; layer counts alone are insufficient}
        close $result
        exit 1
    }
    if {$profile eq "cpu-avx512" && $avxStatus ne "pass"} {
        ::hardwareValidation::row $result profile_config $profile $avxStatus "AVX-512 profile not executed: $avxDetail"
        close $result
        exit 1
    }
    if {![regexp -line {^flags\s*:\s*(.*)$} $cpu -> flags]} {error "CPU flags unavailable"}
    if {$::tcl_platform(machine) ne "x86_64"} {error "x86_64 required"}
    if {$profile in {cpu-avx2 cpu-avx512}} {
        if {$avxState ne "verified"} {error "AVX OS extended state is $avxState"}
        set flagRows 0
        foreach line [split $cpu "\n"] {
            if {![regexp {^flags\s*:\s*(.*)$} $line -> cpuFlags]} {continue}
            incr flagRows
            foreach flag {avx avx2 fma f16c bmi2} {
                if {[lsearch -exact [split $cpuFlags] $flag] < 0} {error "a visible CPU lacks $flag"}
            }
        }
        set processors [regexp -all -line {^processor[ \t]*:} $cpu]
        if {$processors > 0 && $processors != $flagRows} {error {CPU flags missing for one or more visible processors}}
    }
    if {[string trim [exec git -C $source rev-parse HEAD]] ne $pinned || [string trim [exec git -C $source status --porcelain]] ne {}} {error "llama.cpp revision or tree mismatch"}
    set cache [::hardwareValidation::readFile [file join $engine CMakeCache.txt]]
    if {[lsearch -exact [split $cache "\n"] "CMAKE_HOME_DIRECTORY:INTERNAL=$source"] < 0} {error {engine CMake source does not match pinned checkout}}
    if {![regexp -line {^CMAKE_POSITION_INDEPENDENT_CODE:(BOOL|UNINITIALIZED)=ON$} $cache]} {error {engine is not configured for PIC}}
    set cpuOptions {GGML_AVX OFF GGML_AVX2 OFF GGML_FMA OFF GGML_F16C OFF GGML_BMI2 OFF}
    set avx512Option OFF
    if {$profile in {cpu-avx2 cpu-avx512}} {set cpuOptions {GGML_AVX ON GGML_AVX2 ON GGML_FMA ON GGML_F16C ON GGML_BMI2 ON}}
    if {$profile eq "cpu-avx512"} {set avx512Option ON}
    foreach {key expected} [concat {BUILD_SHARED_LIBS OFF GGML_NATIVE OFF GGML_BACKEND_DL OFF GGML_CPU_ALL_VARIANTS OFF GGML_CUDA OFF GGML_HIP OFF GGML_VULKAN OFF GGML_AVX512_VBMI OFF GGML_AVX512_VNNI OFF GGML_AVX512_BF16 OFF GGML_AVX_VNNI OFF GGML_SSE42 OFF} [list GGML_AVX512 $avx512Option] $cpuOptions] {
        if {![regexp -line "^${key}:BOOL=${expected}$" $cache]} {error "engine cache lacks $key=$expected"}
    }
    set compileCommands [::hardwareValidation::readFile [file join $engine compile_commands.json]]
    if {$profile eq "cpu-avx2" && ![regexp -- {-mavx2\M} $compileCommands]} {error "AVX2 compiler flag not found"}
    if {$profile eq "cpu-baseline" && [regexp -- {-m(avx|fma|f16c|bmi)|-march=native} $compileCommands]} {error "baseline contains higher ISA flags"}
    # Verify engine provenance before linking or loading any of its code.
    # An unverifiable build may contain instructions the current CPU cannot run.
    if {[dict get $opts -build-log] eq {}} {
        ::hardwareValidation::row $result profile_config $profile not_run {verbose engine build log not supplied; binding not built or loaded}
        error {engine provenance unavailable}
    }
    if {[catch {::hardwareValidation::verifyEngine $engine $profile [dict get $opts -build-log]} evidence]} {
        ::hardwareValidation::row $result profile_config $profile fail "$evidence; binding not built or loaded"
        error {engine provenance verification failed}
    }
    set configLog [file join [file dirname [dict get $opts -build-log]] llama-cmake.log]
    if {[catch {set configText [::hardwareValidation::readFile $configLog]}] || ![regexp {ggml commit:[[:space:]]+1719dbb0} $configText]} {
        ::hardwareValidation::row $result profile_config $profile fail {configure log missing or does not identify pinned revision; binding not built or loaded}
        error {engine source revision evidence unavailable}
    }
    set archiveHashes {}
    foreach archive {src/libllama.a ggml/src/libggml.a ggml/src/libggml-cpu.a ggml/src/libggml-base.a} {
        set path [file join $engine $archive]
        if {![file isfile $path]} {error "missing static archive: $path"}
        set hash [lindex [exec sha256sum -- $path] 0]
        dict set archiveHashes $archive $hash
        ::hardwareValidation::row $result archive $archive pass "sha256=$hash"
    }
    set fixtures {
        llama 74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45
        minilm 797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662
    }
    foreach {name hash} $fixtures {
        set path [file normalize [file join [dict get $opts -model-store] sha256-$hash]]
        if {![file isfile $path]} {error "model missing: $name"}
        set actual [lindex [exec sha256sum -- $path] 0]
        if {$actual ne $hash} {error "model hash mismatch: $name"}
        dict set models $name $path
        ::hardwareValidation::row $result model $name pass "sha256=$actual"
    }
    set packageDir [file join $out package]
    set run [file join $repo scripts lab run.tcl]
    set runnerResults [file join $out build.tsv]
    set compiler [auto_execok g++]
    if {$compiler eq {}} {error {g++ unavailable}}
    set explicitCompiler "$compiler -march=x86-64 -mtune=generic"
    if {[catch {exec timeout --signal=TERM --kill-after=5s 1200s [info nameofexecutable] $run -repo $repo -llama-src $source -llama-build-dir $engine -build-dir $packageDir -cxx $explicitCompiler -scenarios build -results $runnerResults > [file join $out build-driver.log] 2>@1} message]} {error "binding build failed: $message"}
    set linkLog [::hardwareValidation::readFile [file join $out build-logs build.log]]
    set compileCount 0
    set bindingObjects {}
    foreach line [split $linkLog "\n"] {
        if {[string first { -c } $line] < 0 || [string first $compiler $line] != 0} {continue}
        incr compileCount
        if {[string first {-march=x86-64 -mtune=generic} $line] < 0 || [string first {-fPIC} $line] < 0} {error {binding compiler command lacks baseline or PIC flags}}
        foreach option [regexp -all -inline {\-m[^[:space:]]+} $line] {
            if {$option ni {-march=x86-64 -mtune=generic}} {error "disallowed binding ISA flag: $option"}
        }
        if {![regexp { -o "?([^" ]+)} $line -> object]} {error {binding compiler output is not identifiable}}
        if {![file isfile $object] || [dict exists $bindingObjects $object]} {error "missing or duplicate binding object: $object"}
        dict set bindingObjects $object [lindex [exec sha256sum -- $object] 0]
    }
    if {$compileCount < 1} {error {binding compiler commands absent from build log}}
    set linkLines {}
    foreach line [split $linkLog "\n"] {
        if {[string first {-shared} $line] >= 0 && [string first [file join $packageDir tclllama.so] $line] >= 0} {lappend linkLines $line}
    }
    if {[llength $linkLines] != 1} {error {link log lacks a unique module link invocation}}
    set linkLine [lindex $linkLines 0]
    if {[string first {-march=x86-64 -mtune=generic} $linkLine] < 0} {error {module link lacks explicit CPU flags}}
    dict for {object hash} $bindingObjects {
        if {[string first $object $linkLine] < 0 || [lindex [exec sha256sum -- $object] 0] ne $hash} {error "binding object absent or changed at link: $object"}
    }
    foreach archive {src/libllama.a ggml/src/libggml.a ggml/src/libggml-cpu.a ggml/src/libggml-base.a} {
        if {[string first [file join $engine $archive] $linkLine] < 0} {
            error "link log lacks selected archive: $archive"
        }
        if {[lindex [exec sha256sum -- [file join $engine $archive]] 0] ne [dict get $archiveHashes $archive]} {error "archive changed during link: $archive"}
    }
    ::hardwareValidation::row $result build $profile pass "binding built against selected archives; see build.tsv"
    set module [file normalize [file join $packageDir tclllama.so]]
    set moduleHash [lindex [exec sha256sum -- $module] 0]
    ::hardwareValidation::row $result artifact tclllama.so pass "sha256=$moduleHash"
    # Load in a fresh interpreter and compare its actual module path. The
    # expected path comes from the newly built private package, never auto_path
    # inherited from the caller.
    set loadScript {
        lappend auto_path [lindex $argv 0]
        package require tclllama 0.1.0
        foreach item [info loaded] {
            lassign $item path init
            if {[string equal -nocase $init Tclllama]} {puts [file normalize $path]; exit 0}
        }
        puts stderr "tclllama module not present in info loaded"
        exit 1
    }
    set loadPath [file join $out verify-loaded.tcl]
    set channel [open $loadPath w]
    try {puts $channel $loadScript} finally {close $channel}
    set actual [string trim [exec timeout 30s env -u TCLLIBPATH [info nameofexecutable] $loadPath $packageDir 2>@1]]
    if {[file normalize $actual] ne $module || [lindex [exec sha256sum -- $module] 0] ne $moduleHash} {
        error "loaded module path or SHA-256 does not match built artifact: $actual"
    }
    ::hardwareValidation::row $result artifact loaded_module pass "path=$actual; sha256=$moduleHash"
    ::hardwareValidation::row $result profile_config $profile pass "$evidence; link and loaded module verified"
    set f [open $run r]; set text [read $f]; close $f
    set boundary [string first "\n# Print the supported command-line options." $text]
    if {$boundary < 0} {error "lab runner namespace missing"}
    eval [string range $text 0 [expr {$boundary - 1}]]
    set ::labRun::repoRoot $repo
    foreach suite {compat.test tokenization.test embeddings.test embeddings_batch.test} {
        set embedding [expr {[string match embeddings* $suite] ? [dict get $models minilm] : {}}]
        set decoder [expr {$embedding eq {} ? [dict get $models llama] : {}}]
        set chat [expr {$suite eq "compat.test"}]
        if {![::labRun::testSuite [file join $repo tests $suite] $packageDir [info nameofexecutable] $decoder $embedding $decoder $chat $out $profile]} {set success 0}
    }
    foreach entry $::labRun::rows {
        lassign $entry scenario suite status detail
        ::hardwareValidation::row $result suite "$scenario/$suite" $status $detail
    }
} on error {message} {
    ::hardwareValidation::row $result runner $profile fail $message
    set success 0
} finally {close $result}
exit [expr {$success ? 0 : 1}]
