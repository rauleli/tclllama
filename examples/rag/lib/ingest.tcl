# ingest.tcl -- Bounded embedding batches for the optional RAG SQL stores.
# Provides ::rag::ingest::run; requires Tcl 8.6, a loaded tclllama package and
# one selected ::rag::sqlite or ::rag::mysql store. Input is prefragmented.
# The checkpoint pattern is adapted from tclembedding's sql_ingest.tcl;
# see ../similarity/LICENSE.tclembedding.
package require Tcl 8.6

# Coordinates tclllama embeddings and transactional store checkpoints.
namespace eval ::rag::ingest {}

# Process a list of {record_bytes text} pairs using an existing embedding
# context and a selected store. config must include model hash, tokenizer,
# pooling, normalization, task prefix, fragment policy and source identity.
# Every resumed record is checked against the persisted checksum; embeddings
# use L2 normalization and f32le. SQL commits group up to 64 fragments.
# The binding rejects overlong input and batch
# capacity violations. Errors leave the committed checkpoint available for
# resumption; only a completed corpus becomes searchable.
proc ::rag::ingest::run {backend id config dimension records context prefix batch_size} {
    if {$backend ni {sqlite mysql}} {error "unsupported backend"}
    if {![string is integer -strict $batch_size] || $batch_size < 1 || $batch_size > 32} {error "invalid batch size"}
    set total [llength $records]
    if {$total == 0} {error "empty corpus"}
    foreach record $records {
        if {[llength $record] != 2 || [lindex $record 1] eq ""} {error "invalid fragment"}
    }
    set store ::rag::${backend}
    set store_config [list rag-ingest-v1 $config prefix $prefix dimension $dimension]
    set checkpoint [${store}::begin $id $store_config $dimension $total]
    for {set i 0} {$i < $checkpoint} {incr i} {
        ${store}::verify $id $i [lindex $records $i 0]
    }
    set pending {}
    for {set i $checkpoint} {$i < $total} {} {
        set last [expr {min($total,$i+$batch_size)}]
        set texts {}
        for {set j $i} {$j < $last} {incr j} {
            lappend texts "$prefix[lindex $records $j 1]"
        }
        set vectors [llama::embed_batch $context $texts -normalize l2 -format f32le]
        if {[llength $vectors] != $last-$i} {error "embedding result count mismatch"}
        set rows {}
        for {set j $i} {$j < $last} {incr j} {
            lappend rows [list $j [lindex $records $j 0] [lindex $vectors [expr {$j-$i}]]]
        }
        lappend pending {*}$rows
        set i $last
        if {[llength $pending] >= 64 || $i == $total} {
            set committed [${store}::append $id $pending]
            if {$committed != $i} {error "checkpoint did not advance as expected"}
            set pending {}
            if {$i % 1024 == 0 || $i == $total} {puts stderr "committed_fragments=$i total=$total"; flush stderr}
        }
    }
    ${store}::finish $id
    return $total
}
