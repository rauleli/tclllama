# extract.tcl -- Convert one extractor JSONL record to bounded RAG fragments.
# Provides ::rag::extract::record. Requires rl_json, json::write and the
# fragment module. Inputs and outputs may contain private text; callers must
# store them only in private locations. Adapted from tclembedding's
# examples/03-chatgpt-export/fragment.tcl (see similarity/LICENSE.tclembedding).
package require Tcl 8.6
package require rl_json 0.15.7
package require json::write
namespace eval ::rag::extract {}

# Validate one extractor-v1 record and return {record_bytes text} pairs.
# source_hash identifies the input file, source_id is a stable caller-supplied
# identifier, prefix and budget describe the exact embedding tokenization.
# Offsets use Tcl string indices; every emitted fragment preserves provenance.
proc ::rag::extract::record {line source_id source_hash prefix budget count_command} {
    if {$source_id eq "" || ![regexp {^[0-9a-f]{64}$} $source_hash]} {error "invalid input identity"}
    if {![::rl_json::json valid $line] || [::rl_json::json type $line] ne "object"} {error "invalid input JSON"}
    set keys [::rl_json::json keys $line]
    foreach key {version source conversation_id node_id message_id position role create_time part text} {
        if {$key ni $keys} {error "missing input field: $key"}
    }
    if {[::rl_json::json get $line version] ne "1"} {error "unsupported extractor version"}
    set values {}
    foreach key {source conversation_id node_id message_id role text} {
        if {[::rl_json::json type $line $key] ne "string"} {error "invalid input field: $key"}
        set value [::rl_json::json get $line $key]
        if {$value eq ""} {error "empty input field: $key"}
        dict set values $key $value
    }
    foreach key {position part} {
        if {[::rl_json::json type $line $key] ne "number" || ![string is integer -strict [::rl_json::json get $line $key]] || [::rl_json::json get $line $key] < 0} {error "invalid input field: $key"}
        dict set values $key [::rl_json::json get $line $key]
    }
    if {[::rl_json::json get $line role] ni {user assistant}} {error "unsupported role"}
    if {[::rl_json::json type $line create_time] ni {string number null}} {error "invalid timestamp"}
    set timestamp [::rl_json::json extract $line create_time]
    set chunks [::rag::fragment::split [dict get $values text] $prefix $budget $count_command]
    set output {}
    set index 0
    foreach chunk $chunks {
        lassign $chunk start end text token_count
        set record [::json::write object \
            fragment_version [::json::write string 1] \
            input_source_id [::json::write string $source_id] \
            input_sha256 [::json::write string $source_hash] \
            source [::json::write string [dict get $values source]] \
            conversation_id [::json::write string [dict get $values conversation_id]] \
            node_id [::json::write string [dict get $values node_id]] \
            message_id [::json::write string [dict get $values message_id]] \
            role [::json::write string [dict get $values role]] \
            position [dict get $values position] create_time $timestamp \
            part [dict get $values part] fragment_index $index \
            char_start $start char_end $end token_count $token_count token_budget $budget \
            text [::json::write string $text]]
        lappend output [list [encoding convertto utf-8 $record] $text]
        incr index
    }
    return $output
}
