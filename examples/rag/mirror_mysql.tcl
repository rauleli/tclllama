#!/usr/bin/env tclsh
# mirror_mysql.tcl -- Copy a completed private SQLite corpus to MySQL/InnoDB.
# Usage: sqlite_path mysql_socket database user create|resume corpus_id
# Requires Tcl 8.6, sqlite3, sha256 and tdbc::mysql. No embeddings are computed.
# Both stores preserve per-row SHA-256; only the specified corpus is copied.
# MySQL must be a private dedicated database. Incomplete or corrupt sources are
# rejected, and a failed batch leaves a resumable destination checkpoint.
package require Tcl 8.6
set here [file dirname [file normalize [info script]]]
if {[llength $argv] != 6} {error "usage: sqlite_path socket database user create|resume corpus"}
lassign $argv sqlite_path socket database user action corpus
if {$action ni {create resume}} {error "action must be create or resume"}
source [file join $here lib sqlite_store.tcl]
source [file join $here lib mysql_store.tcl]
set sqlite_open 0
set mysql_open 0
set started 0
try {
    ::rag::sqlite::open $sqlite_path 0
    set sqlite_open 1
    set rows [::rag::sqlite::db eval {SELECT config,dimension,expected FROM corpora WHERE id=$corpus AND status='complete'}]
    if {[llength $rows] != 3} {error "source corpus missing or incomplete"}
    lassign $rows config dimension expected
    ::rag::sqlite::verify_complete $corpus $dimension
    ::rag::mysql::open $socket $database $user
    set mysql_open 1
    if {$action eq "create"} {::rag::mysql::init}
    set checkpoint [::rag::mysql::begin $corpus $config $dimension $expected]
    ::rag::mysql::run {UPDATE rag_corpora SET status='building' WHERE id=:corpus} [list corpus $corpus]
    set started 1
    set batch {}
    ::rag::sqlite::db eval {SELECT sequence,line_data,line_sha,vector,vector_sha FROM fragments WHERE corpus=$corpus ORDER BY sequence} row {
        if {$row(sequence) < $checkpoint} {
            ::rag::mysql::verify $corpus $row(sequence) $row(line_data)
            set saved [::rag::mysql::run {SELECT vector_sha FROM rag_fragments WHERE corpus=:corpus AND sequence_no=:sequence} [list corpus $corpus sequence $row(sequence)]]
            if {[llength $saved] != 1 || [lindex $saved 0 0] ne $row(vector_sha)} {error "destination vector differs from source"}
        } else {
            lappend batch [list $row(sequence) $row(line_data) $row(vector)]
            if {[llength $batch] == 128} {
                set checkpoint [::rag::mysql::append $corpus $batch]
                set batch {}
                puts stderr "mirrored_fragments=$checkpoint total=$expected"
                flush stderr
            }
        }
    }
    if {[llength $batch]} {set checkpoint [::rag::mysql::append $corpus $batch]}
    set target [::rag::mysql::run {SELECT sequence_no,line_sha,vector_sha FROM rag_fragments WHERE corpus=:corpus ORDER BY sequence_no} [list corpus $corpus]]
    set index 0
    ::rag::sqlite::db eval {SELECT sequence,line_sha,vector_sha FROM fragments WHERE corpus=$corpus ORDER BY sequence} row {
        if {[lindex $target $index] ne [list $row(sequence) $row(line_sha) $row(vector_sha)]} {error "source and destination differ"}
        incr index
    }
    if {$index != $expected || [llength $target] != $expected} {error "source and destination counts differ"}
    ::rag::mysql::finish $corpus
    ::rag::mysql::verify_complete $corpus $dimension
    puts "mirrored_fragments=$index status=complete equal_hashes=yes"
} on error {message options} {
    if {$started} {
        catch {::rag::mysql::run {UPDATE rag_corpora SET status='building' WHERE id=:corpus} [list corpus $corpus]}
    }
    return -options $options $message
} finally {
    if {$mysql_open} {catch {::rag::mysql::close}}
    if {$sqlite_open} {catch {::rag::sqlite::close}}
}
