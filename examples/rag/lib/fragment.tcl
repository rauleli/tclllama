# fragment.tcl -- Token-budget fragmentation using the active GGUF tokenizer.
# Provides ::rag::fragment::split. Requires Tcl 8.6; caller provides a token
# count callback, for example one invoking llama::tokenize on an embedding
# context. Adapted conceptually from tclembedding's fragment.tcl; its license
# is included at ../similarity/LICENSE.tclembedding.
package require Tcl 8.6
namespace eval ::rag::fragment {}

# Return the next Unicode boundary after offset in Tcl string units. On Tcl
# 8.6 supplementary characters can occupy two units; preserve surrogate pairs.
proc ::rag::fragment::next_boundary {text offset} {
    set size [string length $text]
    if {$offset >= $size} {return $size}
    scan [string index $text $offset] %c code
    set next [expr {$offset+1}]
    if {$code >= 0xd800 && $code <= 0xdbff && $next < $size} {
        scan [string index $text $next] %c low
        if {$low >= 0xdc00 && $low <= 0xdfff} {incr next}
    }
    return $next
}

# Split text into ordered {start end text token_count} fragments with exclusive
# end offsets. count_command receives the complete prefixed text and returns a
# nonnegative token count, including any special tokens. Each emitted fragment
# has been measured within budget; concatenating fragment texts recovers input.
# The algorithm probes increasing boundaries and makes no optimality claim.
# It errors if even one character cannot fit; it never silently truncates.
proc ::rag::fragment::split {text prefix budget count_command} {
    if {![string is integer -strict $budget] || $budget < 1} {error "invalid token budget"}
    if {$text eq ""} {error "empty input"}
    set size [string length $text]
    set start 0
    set result {}
    while {$start < $size} {
        set end [next_boundary $text $start]
        set best $start
        set best_count 0
        set step 1
        while {1} {
            set candidate [string range $text $start [expr {$end-1}]]
            set count [{*}$count_command "$prefix$candidate"]
            if {![string is integer -strict $count] || $count < 0} {error "invalid tokenizer count"}
            if {$count > $budget} {break}
            set best $end
            set best_count $count
            if {$end == $size} {break}
            set target [expr {min($size,$end+$step)}]
            while {$end < $target} {set end [next_boundary $text $end]}
            set step [expr {min(1024,$step*2)}]
        }
        if {$best == $start} {error "token budget cannot fit next character"}
        lappend result [list $start $best [string range $text $start [expr {$best-1}]] $best_count]
        set start $best
    }
    return $result
}

# Count tokens with the exact options used by llama::embed and embed_batch.
proc ::rag::fragment::llama_count {context text} {
    return [llength [llama::tokenize $context $text -add_special true -parse_special false]]
}
