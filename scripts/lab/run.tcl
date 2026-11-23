#!/usr/bin/env tclsh
# -----------------------------------------------------------------------------
# run.tcl -- Configurable tclllama lab runner
#
# Purpose: build the package, inspect a host, run existing Tcl/native suites,
#          and invoke the established embedding benchmark with supplied paths.
# Public interface: options shown by `tclsh run.tcl -help`; results are TSV.
# Requirements: Tcl 8.6, make, project sources and caller-selected local models.
# No models are downloaded and absent GPU/ISA features are recorded as skips.
# -----------------------------------------------------------------------------

namespace eval ::labRun {
    variable config
    variable rows {}
    variable repoRoot {}

    # Record one result with the shared status vocabulary. Detail is sanitized
    # to keep the output a valid one-record-per-line TSV file.
    proc record {scenario target status detail} {
        variable rows
        variable repoRoot
        set replacements [list "\t" { } "\r" { } "\n" { }]
        if {$repoRoot ne {}} {lappend replacements $repoRoot {$REPO}}
        if {[info exists ::env(HOME)] && $::env(HOME) ne {}} {
            lappend replacements $::env(HOME) {~}
        }
        set cleanTarget [string map $replacements $target]
        set cleanDetail [string map $replacements $detail]
        lappend rows [list $scenario $cleanTarget $status $cleanDetail]
    }

    # Run a command and save its output in a per-case log. Benchmark callers
    # can save stderr separately. Returns success without invoking a shell.
    proc command {scenario target logPath argv {separateStderr 0}} {
        if {$separateStderr} {
            set redirect [list 2> "${logPath}.stderr"]
        } else {
            set redirect [list 2>@1]
        }
        if {[catch {exec {*}$argv > $logPath {*}$redirect} message]} {
            record $scenario $target fail $message
            return 0
        }
        record $scenario $target pass "log=[file tail $logPath]"
        return 1
    }

    # Resolve a model from the configured Ollama-style SHA-256 blob store. The
    # digest is supplied by the project manifest, avoiding filename searches.
    proc blobPath {store digest} {
        return [file join $store "sha256-$digest"]
    }

    # Invoke one Tcltest suite with explicit model variables and save its full
    # output. Chat tests are enabled only for the generative model runs.
    proc testSuite {testPath packageDir tclsh model embedding decoder chat logDir label {extraEnv {}} {timeoutSeconds 300}} {
        set suite [file rootname [file tail $testPath]]
        set logPath [file join $logDir "$label-$suite.log"]
        set envCommand [list env "TCLLIBPATH=$packageDir"]
        foreach {key value} [list \
            TCLLLAMA_TEST_MODEL $model \
            TCLLLAMA_TEST_EMBEDDING_MODEL $embedding \
            TCLLLAMA_TEST_DECODER_MODEL $decoder \
            TCLLLAMA_TEST_CHAT $chat] {
            # Set blank values explicitly so inherited model paths cannot leak
            # into scenarios that intentionally omit a decoder or embedding.
            lappend envCommand "$key=$value"
        }
        foreach {key value} $extraEnv {lappend envCommand "$key=$value"}
        # Bound each suite; absence of timeout is a runner error rather than an
        # unbounded process or an apparent suite pass.
        if {[auto_execok timeout] eq {}} {
            record $label $suite fail {timeout utility unavailable; suite not executed}
            return 0
        }
        if {![string is integer -strict $timeoutSeconds] || $timeoutSeconds < 1} {
            record $label $suite fail {invalid suite timeout}
            return 0
        }
        lappend envCommand timeout --signal=TERM --kill-after=5s ${timeoutSeconds}s $tclsh $testPath $packageDir
        if {[catch {exec {*}$envCommand > $logPath 2>@1} message]} {
            record $label $suite fail "exit error; log=[file tail $logPath]; $message"
            return 0
        }
        if {[catch {set input [open $logPath r]; set output [read $input]; close $input} message]} {
            record $label $suite fail "cannot read log: $message"
            return 0
        }
        set pattern {(?n)^([^\r\n]*)Total[ \t]+([0-9]+)[ \t]+Passed[ \t]+([0-9]+)[ \t]+Skipped[ \t]+([0-9]+)[ \t]+Failed[ \t]+([0-9]+)[ \t]*\r?$}
        set matches [regexp -all -inline $pattern $output]
        if {[llength $matches] != 6} {
            record $label $suite fail "missing or ambiguous Tcltest summary; log=[file tail $logPath]"
            return 0
        }
        lassign $matches summary prefix total passed skipped failed
        if {$total != $passed + $skipped + $failed || $failed > 0} {
            record $label $suite fail "$summary; invalid counts or failures; log=[file tail $logPath]"
            return 0
        }
        if {$passed == 0} {
            record $label $suite skip "$summary; no cases passed; log=[file tail $logPath]"
            return 0
        }
        record $label $suite pass "log=[file tail $logPath]; $summary"
        return 1
    }

    # Append collected records using the lab result schema.
    proc writeResults {path} {
        variable rows
        set channel [open $path a]
        try {
            foreach row $rows {puts $channel [join $row "\t"]}
        } finally {
            close $channel
        }
    }
}

# Print the supported command-line options. Called for `-help` and parse errors.
proc usage {} {
    puts stderr {Usage: tclsh run.tcl ?options?
  -repo PATH            repository root (default: detected from script)
  -model-store PATH     SHA-256 blob directory
  -build-dir PATH       package build directory
  -test-build-dir PATH  separate TCLLLAMA_TESTING package directory
  -llama-src PATH       pinned llama.cpp checkout
  -llama-build-dir PATH llama.cpp CMake output used by Makefile.local
  -engine-options LIST  extra explicit CMake options (engine-build scenario)
  -tclsh PATH           Tcl executable (default: current executable)
  -cc PATH              C compiler for the pinned engine CMake build
  -cxx PATH             C++ compiler for make
  -cxxflags FLAGS       compile flags as one make variable value
  -scenarios LIST       inspect,engine-build,build,regression,embeddings,testing,native,sanitizers,benchmark
  -repetitions N        embedding benchmark repetitions (minimum 5)
  -threads N            benchmark threads (default: 2)
  -results PATH         TSV output (default: ../experimentos/laboratorio-055/gepeto.tsv)
  -model PATH           generative model override
  -qwen-model PATH      second generative model override
  -embedding-model PATH embedding model override (default: MiniLM blob)
  -decoder-model PATH   decoder embedding model override (default: Qwen blob)
  -help                 show this message

The runner does not download models or infer hardware support from another
feature. Missing models/builds are reported as skip or unsupported.}
}

set scriptDir [file dirname [file normalize [info script]]]
set repoDefault [file normalize [file join $scriptDir ../..]]
set options [dict create \
    -repo $repoDefault \
    -model-store {} \
    -llama-src {} \
    -llama-build-dir {} \
    -engine-options {} \
    -build-dir {} \
    -test-build-dir {} \
    -tclsh [info nameofexecutable] \
    -cc cc \
    -cxx c++ \
    -cxxflags -O2 \
    -scenarios {inspect build regression embeddings testing native sanitizers benchmark} \
    -repetitions 5 \
    -threads 2 \
    -results {} \
    -model {} \
    -qwen-model {} \
    -embedding-model {} \
    -decoder-model {}]
if {[lsearch -exact $argv -help] >= 0} {usage; exit 0}
if {$argc % 2} {usage; exit 2}
foreach {option value} $argv {
    if {![dict exists $options $option]} {puts stderr "Unknown option: $option"; usage; exit 2}
    dict set options $option $value
}

dict set options -repo [file normalize [dict get $options -repo]]
set repo [dict get $options -repo]
set ::labRun::repoRoot $repo
cd $repo
if {[dict get $options -results] eq {}} {
    dict set options -results [file normalize [file join $repo ../experimentos/laboratorio-055/gepeto.tsv]]
} else {
    dict set options -results [file normalize [dict get $options -results]]
}
if {[dict get $options -model-store] eq {}} {
    dict set options -model-store [file normalize [file join $repo ../datos/modelos/blobs]]
} else {
    dict set options -model-store [file normalize [dict get $options -model-store]]
}
foreach key {-build-dir -test-build-dir} {
    if {[dict get $options $key] eq {}} {
        set suffix [expr {$key eq "-test-build-dir" ? "audit055-testing" : "audit055"}]
        dict set options $key [file join $repo build $suffix]
    }
    dict set options $key [file normalize [dict get $options $key]]
}
if {[dict get $options -llama-src] eq {}} {
    dict set options -llama-src [file normalize [file join $repo ../dependencias/llama.cpp]]
} else {
    dict set options -llama-src [file normalize [dict get $options -llama-src]]
}
if {[dict get $options -llama-build-dir] eq {}} {
    dict set options -llama-build-dir [file join [dict get $options -llama-src] build-1719dbb-static-pic]
} else {
    dict set options -llama-build-dir [file normalize [dict get $options -llama-build-dir]]
}
set model [dict get $options -model]
if {$model eq {}} {set model [::labRun::blobPath [dict get $options -model-store] 74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45]}
set qwenModel [dict get $options -qwen-model]
if {$qwenModel eq {}} {set qwenModel [::labRun::blobPath [dict get $options -model-store] 3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6]}
set embedding [dict get $options -embedding-model]
if {$embedding eq {}} {set embedding [::labRun::blobPath [dict get $options -model-store] 797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662]}
set decoder [dict get $options -decoder-model]
if {$decoder eq {}} {set decoder [::labRun::blobPath [dict get $options -model-store] 970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6]}
set packageDir [dict get $options -build-dir]
set resultPath [dict get $options -results]
file mkdir [file dirname $resultPath]
set resultChannel [open $resultPath w]
puts $resultChannel "scenario\ttarget\tstatus\tdetail"
close $resultChannel
set logDir [file join [file dirname $resultPath] [file rootname [file tail $resultPath]]-logs]
file mkdir $logDir
set success 1
foreach {key value} [list \
    repository {$REPO} \
    model_store {configured model store} \
    build_dir [file tail $packageDir] \
    llama_build_dir [file tail [dict get $options -llama-build-dir]] \
    tclsh [file tail [dict get $options -tclsh]] \
    compiler [file tail [dict get $options -cxx]] \
    compiler_flags [dict get $options -cxxflags] \
    llama_model [file tail $model] \
    qwen_model [file tail $qwenModel] \
    embedding_model [file tail $embedding] \
    decoder_embedding_model [file tail $decoder] \
    threads [dict get $options -threads] \
    repetitions [dict get $options -repetitions] \
    llama_revision 1719dbb0ee7bdaf56de083079d3613bb670160a5] {
    ::labRun::record config $key pass $value
}

foreach scenario [dict get $options -scenarios] {
    switch -- $scenario {
        engine-build {
            set engineBuild [dict get $options -llama-build-dir]
            set cmakeCommand [list cmake -S [dict get $options -llama-src] -B $engineBuild \
                -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=ON \
                -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX_VNNI=OFF \
                -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
                -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \
                -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
                "-DCMAKE_C_COMPILER=[dict get $options -cc]" \
                "-DCMAKE_CXX_COMPILER=[dict get $options -cxx]"]
            foreach extraOption [dict get $options -engine-options] {
                lappend cmakeCommand $extraOption
            }
            if {[::labRun::command engine-build configure [file join $logDir llama-cmake.log] $cmakeCommand]} {
                # Preserve the actual compiler and archiver invocations for
                # later inspection of the engine build, not just CMake cache.
                set compileCommand [list cmake --build $engineBuild --parallel 2 --verbose]
                if {![::labRun::command engine-build compile [file join $logDir llama-build.log] $compileCommand]} {
                    set success 0
                }
            } else {
                set success 0
            }
        }
        inspect {
            set inspectPath [file join $repo scripts lab inspect.tcl]
            set inspectOutput [file join [file dirname $resultPath] environment.tsv]
            if {[catch {exec [dict get $options -tclsh] $inspectPath \
                -package-dir $packageDir -output $inspectOutput 2>@1} message]} {
                ::labRun::record inspect host fail $message
                set success 0
            } else {
                ::labRun::record inspect host pass "report=$inspectOutput"
            }
        }
        build {
            set command [list make -f Makefile.local "LLAMA_DIR=[dict get $options -llama-src]" \
                "LLAMA_BUILD=[dict get $options -llama-build-dir]" "BUILD_DIR=$packageDir" \
                "CXX=[dict get $options -cxx]" "CXXFLAGS=[dict get $options -cxxflags]" all]
            if {![::labRun::command build package [file join $logDir build.log] \
                [list {*}$command]]} {set success 0}
        }
        regression {
            if {![file isfile [file join $packageDir tclllama.so]]} {
                ::labRun::record regression package unsupported {build directory has no tclllama.so; use scenario build}
                set success 0
            } else {
                foreach fixture [list [list llama $model] [list qwen $qwenModel]] {
                    lassign $fixture label fixturePath
                    if {![file isfile $fixturePath]} {
                        ::labRun::record regression $label skip "model blob unavailable: $fixturePath"
                        set success 0
                        continue
                    }
                    foreach testPath [lsort [glob -nocomplain [file join $repo tests *.test]]] {
                        if {![::labRun::testSuite $testPath $packageDir \
                            [dict get $options -tclsh] $fixturePath $embedding $fixturePath 1 \
                            $logDir $label]} {set success 0}
                    }
                }
                foreach fixture [list [list minilm $embedding] [list nomic $decoder]] {
                    lassign $fixture label fixturePath
                    if {![file isfile $fixturePath]} {
                        ::labRun::record regression $label skip "model blob unavailable: $fixturePath"
                        set success 0
                        continue
                    }
                    foreach testName {embeddings.test embeddings_batch.test} {
                        set testPath [file join $repo tests $testName]
                        if {![::labRun::testSuite $testPath $packageDir \
                            [dict get $options -tclsh] {} $fixturePath {} 0 \
                            $logDir $label]} {set success 0}
                    }
                }
            }
        }
        embeddings {
            if {![file isfile [file join $packageDir tclllama.so]]} {
                ::labRun::record embeddings package unsupported {build directory has no tclllama.so; use scenario build}
                set success 0
            } else {
                foreach fixture [list \
                    [list minilm $embedding] \
                    [list nomic $decoder]] {
                    lassign $fixture label fixturePath
                    if {![file isfile $fixturePath]} {
                        ::labRun::record embeddings $label skip "model blob unavailable: [file tail $fixturePath]"
                    } else {
                        foreach testName {embeddings.test embeddings_batch.test} {
                            set testPath [file join $repo tests $testName]
                            if {![::labRun::testSuite $testPath $packageDir \
                                [dict get $options -tclsh] {} $fixturePath {} 0 \
                                $logDir $label]} {set success 0}
                        }
                    }
                }
            }
        }
        testing {
            set testingDir [dict get $options -test-build-dir]
            set testFlags "[dict get $options -cxxflags] -DTCLLLAMA_TESTING"
            set command [list make -f Makefile.local "LLAMA_DIR=[dict get $options -llama-src]" \
                "LLAMA_BUILD=[dict get $options -llama-build-dir]" "BUILD_DIR=$testingDir" \
                "CXX=[dict get $options -cxx]" "CXXFLAGS=$testFlags" all]
            if {![::labRun::command testing build [file join $logDir testing-build.log] $command]} {
                set success 0
            } else {
                foreach fixture [list \
                    [list llama $model] \
                    [list qwen $qwenModel]] {
                    lassign $fixture label fixturePath
                    foreach testName {failure_injection.test shared_model.test embeddings_batch.test} {
                        set testPath [file join $repo tests $testName]
                        if {![file isfile $fixturePath]} {
                            ::labRun::record testing $testName skip "model blob unavailable: $fixturePath"
                        } elseif {![::labRun::testSuite $testPath $testingDir \
                            [dict get $options -tclsh] $fixturePath $embedding $fixturePath 0 \
                            $logDir "testing-$label"]} {
                            set success 0
                        }
                    }
                }
            }
        }
        sanitizers {
            set sanitizerDir [file join $repo build audit055-asan]
            set compileFlags {-O1 -g -fno-omit-frame-pointer -fsanitize=address -DTCLLLAMA_TESTING}
            set command [list make -f Makefile.local "LLAMA_DIR=[dict get $options -llama-src]" \
                "LLAMA_BUILD=[dict get $options -llama-build-dir]" "BUILD_DIR=$sanitizerDir" \
                "CXX=[dict get $options -cxx] -fsanitize=address" \
                "CXXFLAGS=$compileFlags" all]
            if {![::labRun::command sanitizers build [file join $logDir asan-build.log] $command]} {
                set success 0
            } else {
                set asanRuntime {}
                if {[catch {set asanRuntime [string trim [exec [dict get $options -cxx] -print-file-name=libasan.so.8]]}]} {
                    set asanRuntime {}
                }
                if {$asanRuntime eq {} || ![file isfile $asanRuntime]} {
                    ::labRun::record sanitizers libasan unsupported $asanRuntime
                } else {
                    set preload "$asanRuntime:[expr {[info exists ::env(LD_PRELOAD)] ? $::env(LD_PRELOAD) : {}}]"
                    set extraEnv [list \
                        LD_PRELOAD [string trimright $preload :] \
                        ASAN_OPTIONS detect_leaks=1:halt_on_error=1:alloc_dealloc_mismatch=1:strict_string_checks=1]
                    foreach fixture [list \
                        [list llama $model] \
                        [list qwen $qwenModel]] {
                        lassign $fixture label fixturePath
                        foreach testName {lifecycle.test failure_injection.test shared_model.test sequence_text.test streaming.test grammar.test} {
                            set testPath [file join $repo tests $testName]
                            if {[file isfile $fixturePath]} {
                                if {![::labRun::testSuite $testPath $sanitizerDir \
                                    [dict get $options -tclsh] $fixturePath {} {} 1 $logDir "asan-$label" $extraEnv]} {
                                    set success 0
                                }
                            }
                        }
                    }
                    foreach fixture [list \
                        [list minilm $embedding] \
                        [list nomic $decoder]] {
                        lassign $fixture label fixturePath
                        foreach testName {embeddings.test embeddings_batch.test} {
                            set testPath [file join $repo tests $testName]
                            if {[file isfile $fixturePath] && ![::labRun::testSuite $testPath $sanitizerDir \
                                [dict get $options -tclsh] {} $fixturePath {} 0 $logDir "asan-$label" $extraEnv]} {
                                set success 0
                            }
                        }
                    }
                }
            }
        }
        native {
            foreach target {sampler-history-test logit-bias-test sampling-strategies-test schema-adapter-test gguf-fixture} {
                set command [list make -f Makefile.local "LLAMA_DIR=[dict get $options -llama-src]" \
                    "LLAMA_BUILD=[dict get $options -llama-build-dir]" "BUILD_DIR=$packageDir" $target]
                if {![::labRun::command native $target [file join $logDir "$target.log"] $command]} {set success 0}
            }
            set streamTest [file join $logDir stream-buffer-test]
            if {[catch {exec c++ -std=c++11 [file join $repo tests stream_buffer.cpp] -o $streamTest 2>@1} message]} {
                ::labRun::record native stream_buffer fail $message
                set success 0
            } elseif {![::labRun::command native stream_buffer [file join $logDir stream_buffer.log] [list $streamTest]]} {
                set success 0
            }
            if {[file isfile [file join $packageDir tclllama.so]]} {
                set testPath [file join $repo tests gguf.test]
                if {![::labRun::testSuite $testPath $packageDir [dict get $options -tclsh] {} {} {} 0 $logDir native]} {set success 0}
            }
            if {![::labRun::command native registry [file join $logDir registry.log] \
                [list [dict get $options -tclsh] [file join $repo tests registry.test]]]} {set success 0}
        }
        benchmark {
            if {![file isfile [file join $packageDir tclllama.so]]} {
                ::labRun::record benchmark package unsupported {build directory has no tclllama.so; use scenario build}
                set success 0
            } else {
                foreach fixture [list \
                    [list llama $model] \
                    [list qwen $qwenModel]] {
                    lassign $fixture label fixturePath
                    set logPath [file join $logDir "benchmark-$label-generation.tsv"]
                    if {![file isfile $fixturePath]} {
                        ::labRun::record benchmark $label skip "model blob unavailable: $fixturePath"
                        set success 0
                    } else {
                        set command [list [dict get $options -tclsh] \
                            [file join $repo scripts lab generation_benchmark.tcl] \
                            $packageDir $fixturePath $label [dict get $options -repetitions] \
                            [dict get $options -threads]]
                        if {![::labRun::command benchmark "$label-generation" $logPath $command 1]} {set success 0}
                    }
                }
                foreach fixture [list \
                    [list all-minilm $embedding {}] \
                    [list nomic $decoder {search_document: }]] {
                    lassign $fixture label fixturePath prefix
                    set logPath [file join $logDir "benchmark-$label.tsv"]
                    if {![file isfile $fixturePath]} {
                        ::labRun::record benchmark $label skip "model blob unavailable: $fixturePath"
                        set success 0
                    } else {
                        set command [list [dict get $options -tclsh] \
                            [file join $repo tests embeddings_benchmark.tcl] \
                            $packageDir $fixturePath $label [dict get $options -repetitions] \
                            $prefix [dict get $options -threads]]
                        if {![::labRun::command benchmark $label $logPath $command 1]} {set success 0}
                    }
                }
            }
        }
        default {
            ::labRun::record options $scenario unsupported {unknown scenario; use -help}
            set success 0
        }
    }
    ::labRun::writeResults $resultPath
    set ::labRun::rows {}
}

if {!$success} {exit 1}
exit 0
