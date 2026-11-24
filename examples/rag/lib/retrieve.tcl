# retrieve.tcl -- Shared semantic retrieval for the optional RAG examples.
# Provides ::rag::retrieve::query. Requires Tcl 8.6, rl_json, tclllama and a
# previously opened SQLite or MySQL RAG store. Input records are UTF-8 JSON.
package require Tcl 8.6
package require rl_json 0.15.7

# Shared retrieval returns evidence text and caller-visible source metadata.
namespace eval ::rag::retrieve {}

# Embed the query using tclllama and search the selected completed corpus.
# Records must contain text, conversation_id, message_id and fragment_index.
# min_score is an explicit application threshold; scores from different models
# must not be compared. Invalid records and backend errors are propagated.
proc ::rag::retrieve::query {backend corpus expected_config context prefix min_score question topk} {
    if {$backend ni {sqlite mysql}} {error "unsupported backend"}
    if {![string is double -strict $min_score] || [catch {expr {double($min_score)==double($min_score) && abs(double($min_score)) <= 1.0}} valid] || !$valid} {error "invalid score threshold"}
    ::rag::${backend}::check_config $corpus $expected_config
    set vector [llama::embed $context "$prefix$question" -normalize l2 -format f32le]
    set matches [::rag::${backend}::search $corpus $vector $topk]
    set result {}
    foreach match $matches {
        lassign $match sequence line score
        if {$score < $min_score} {continue}
        set json [encoding convertfrom utf-8 $line]
        if {![::rl_json::json valid $json] || [::rl_json::json type $json] ne "object"} {error "invalid stored record"}
        foreach key {text conversation_id message_id fragment_index} {
            if {$key ni [::rl_json::json keys $json]} {error "stored record lacks provenance"}
        }
        set text [::rl_json::json get $json text]
        set conversation [::rl_json::json get $json conversation_id]
        set message [::rl_json::json get $json message_id]
        set fragment [::rl_json::json get $json fragment_index]
        lappend result [dict create text $text source [list conversation $conversation message $message fragment $fragment sequence $sequence score $score]]
    }
    return $result
}
