# generate.tcl -- Native and local Ollama chat callbacks for the RAG REPL.
# Provides ::rag::generate::native and ::rag::generate::ollama. Tcl 8.6,
# tclllama or Tcl http, json::write and rl_json are required as appropriate.
package require Tcl 8.6
namespace eval ::rag::generate {}

# Generate one response with an existing tclllama context. The caller supplies
# the complete messages, a sampling dictionary and a positive token limit.
proc ::rag::generate::native {context options max_tokens messages} {
    set result [llama::chat $context $messages -options $options -max_tokens $max_tokens -details true]
    set answer [dict get $result output]
    set reason [dict get $result reason]
    if {$reason in {token_limit context_limit cancelled}} {
        append answer "\n\[Response incomplete: $reason\]"
    }
    return $answer
}

# Generate one nonstreaming response using an Ollama server on loopback IPv4.
# HTTP timeouts, non-success status, incomplete responses and invalid JSON are
# errors. Requests explicitly disable Ollama thinking when supported. Model
# management and all embeddings remain outside this callback.
proc ::rag::generate::ollama {port model timeout_ms max_tokens messages} {
    package require http
    package require json::write
    package require rl_json 0.15.7
    if {![string is integer -strict $port] || $port < 1 || $port > 65535 || ![string is integer -strict $timeout_ms] || $timeout_ms < 1 || ![string is integer -strict $max_tokens] || $max_tokens < 1 || $model eq ""} {error "invalid Ollama configuration"}
    set items {}
    foreach message $messages {
        lappend items [::json::write object role [::json::write string [dict get $message role]] content [::json::write string [dict get $message content]]]
    }
    set options [::json::write object temperature 0.2 top_k 40 top_p 0.9 num_ctx 2048 num_thread 2 num_predict $max_tokens]
    set body [::json::write object model [::json::write string $model] stream false think false options $options messages [::json::write array {*}$items]]
    set token [::http::geturl "http://127.0.0.1:$port/api/chat" -method POST -type application/json -query [encoding convertto utf-8 $body] -timeout $timeout_ms]
    try {
        if {[::http::status $token] ne "ok"} {error "Ollama request failed: [::http::status $token]"}
        if {[::http::ncode $token] != 200} {error "Ollama HTTP status [::http::ncode $token]"}
        set data [encoding convertfrom utf-8 [::http::data $token]]
        if {![::rl_json::json valid $data] || [::rl_json::json type $data] ne "object"} {error "invalid Ollama JSON"}
        if {[catch {::rl_json::json type $data done} type] || $type ne "boolean" || ![::rl_json::json get $data done]} {error "incomplete Ollama response"}
        if {[catch {::rl_json::json get $data message content} answer] || $answer eq ""} {error "missing Ollama response content"}
        if {[catch {::rl_json::json get $data done_reason} reason]} {
            append answer "\n\[Completion reason unavailable\]"
        } elseif {$reason eq "length"} {
            append answer "\n\[Response incomplete: token limit\]"
        }
        return $answer
    } finally {
        ::http::cleanup $token
    }
}
