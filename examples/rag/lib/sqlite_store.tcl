# sqlite_store.tcl -- SQLite persistence for RAG fragments and f32le vectors.
# Provides ::rag::sqlite procedures; requires Tcl 8.6, sqlite3 and sha256.
# Adapted from tclembedding's examples/03-chatgpt-export/sqlite_retrieval.tcl;
# see ../similarity/LICENSE.tclembedding. Databases contain private data.
package require Tcl 8.6
package require sqlite3
package require sha256

# One open store per namespace; callers close it before opening another.
namespace eval ::rag::sqlite {
    variable public_root [file normalize [file join [file dirname [info script]] .. .. ..]]
}

# Validate f32le vector shape, finiteness and nonzero norm before SQL use.
proc ::rag::sqlite::validate_vector {vector dimension} {
    if {[string length $vector] != 4*$dimension} {error "vector dimension mismatch"}
    binary scan $vector r* values
    set norm 0.0
    foreach value $values {
        if {[catch {expr {double($value)==double($value) && abs(double($value)) < Inf}} finite] || !$finite} {error "non-finite vector"}
        set norm [expr {$norm+double($value)*double($value)}]
    }
    if {$norm == 0.0} {error "zero vector"}
}

# Return a SHA-256 hexadecimal digest for a byte string.
proc ::rag::sqlite::digest {bytes} {
    return [::sha2::sha256 -hex $bytes]
}

# Open an existing store or create a new empty one exclusively. The caller
# supplies a private path; failure to initialize removes no existing files.
proc ::rag::sqlite::open {path create} {
    variable public_root
    set path [file normalize $path]
    if {$path eq $public_root || [string first "$public_root/" "$path/"] == 0} {error "database must be outside the public source tree"}
    if {[llength [info commands ::rag::sqlite::db]]} {
        error "store already open"
    }
    if {$create} {
        set f [::open $path {WRONLY CREAT EXCL} 0600]
        ::close $f
    } elseif {![file isfile $path]} {
        error "database does not exist"
    }
    sqlite3 ::rag::sqlite::db $path -create false
    db eval {PRAGMA foreign_keys=ON}
    if {$create} {
        db transaction {
            db eval {
                PRAGMA user_version=1;
                CREATE TABLE corpora (
                    id TEXT PRIMARY KEY, config TEXT NOT NULL,
                    config_sha TEXT NOT NULL, dimension INTEGER NOT NULL,
                    expected INTEGER NOT NULL, progress INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('building','complete'))
                );
                CREATE TABLE fragments (
                    corpus TEXT NOT NULL, sequence INTEGER NOT NULL,
                    line_data BLOB NOT NULL, line_sha TEXT NOT NULL,
                    vector BLOB NOT NULL, vector_sha TEXT NOT NULL,
                    PRIMARY KEY(corpus,sequence),
                    FOREIGN KEY(corpus) REFERENCES corpora(id)
                );
            }
        }
    }
    if {[db onecolumn {PRAGMA user_version}] != 1} {
        db close
        error "unsupported database schema"
    }
}

# Close the current connection and release SQLite resources.
proc ::rag::sqlite::close {} {
    db close
}

# Create or resume a corpus with immutable identity, model configuration,
# dimension and expected count. Returns the verified committed checkpoint.
proc ::rag::sqlite::begin {id config dimension expected} {
    if {![string is integer -strict $dimension] || $dimension ni {384 768} ||
        ![string is integer -strict $expected] || $expected < 1} {
        error "invalid dimension or expected count"
    }
    set config_sha [digest [encoding convertto utf-8 $config]]
    db transaction {
        db eval {INSERT OR IGNORE INTO corpora VALUES($id,$config,$config_sha,$dimension,$expected,0,'building')}
        set n [db onecolumn {SELECT count(*) FROM corpora WHERE id=$id AND config=$config AND config_sha=$config_sha AND dimension=$dimension AND expected=$expected}]
        if {$n != 1} {error "corpus configuration conflict"}
        set progress [db onecolumn {SELECT progress FROM corpora WHERE id=$id}]
        set count [db onecolumn {SELECT count(*) FROM fragments WHERE corpus=$id}]
        set last [db onecolumn {SELECT max(sequence) FROM fragments WHERE corpus=$id}]
        if {$progress != $count || ($progress > 0 && $last != $progress-1)} {
            error "checkpoint mismatch"
        }
    }
    return $progress
}

# Atomically append contiguous rows {sequence utf8_record vector}; a failure
# rolls back both rows and checkpoint. Callers provide complete metadata records.
proc ::rag::sqlite::append {id rows} {
    if {![llength $rows]} {error "empty batch"}
    db transaction {
        set state [db eval {SELECT progress,dimension,expected FROM corpora WHERE id=$id AND status='building'}]
        if {[llength $state] != 3} {error "corpus is not building"}
        lassign $state progress dimension expected
        foreach row $rows {
            if {[llength $row] != 3} {error "invalid row"}
            lassign $row sequence line vector
            if {$sequence != $progress || $sequence >= $expected || [string length $vector] != 4*$dimension} {
                error "invalid sequence or vector dimension"
            }
            binary scan $vector r* values
            set norm 0.0
            foreach v $values {
                if {[catch {expr {double($v)==double($v) && abs(double($v)) < Inf}} finite] || !$finite} {
                    error "non-finite vector"
                }
                set norm [expr {$norm+double($v)*double($v)}]
            }
            if {$norm == 0.0} {error "zero vector"}
            set line_sha [digest $line]
            set vector_sha [digest $vector]
            db eval {INSERT INTO fragments VALUES($id,$sequence,$line,$line_sha,$vector,$vector_sha)}
            incr progress
        }
        db eval {UPDATE corpora SET progress=$progress WHERE id=$id}
    }
    return $progress
}

# Verify hashes and input bytes of a previously committed row before resuming.
proc ::rag::sqlite::verify {id sequence line} {
    set found 0
    db eval {SELECT line_data,line_sha,vector,vector_sha FROM fragments WHERE corpus=$id AND sequence=$sequence} row {
        set found 1
        if {$row(line_data) ne $line || [digest $row(line_data)] ne $row(line_sha) ||
            [digest $row(vector)] ne $row(vector_sha)} {error "stored row mismatch"}
    }
    if {!$found} {error "missing stored row"}
}

# Mark a complete corpus available for querying after checking its checkpoint.
proc ::rag::sqlite::finish {id} {
    db transaction {
        set n [db onecolumn {SELECT count(*) FROM corpora WHERE id=$id AND expected=progress AND progress=(SELECT count(*) FROM fragments WHERE corpus=$id)}]
        if {$n != 1} {error "incomplete corpus"}
        db eval {UPDATE corpora SET status='complete' WHERE id=$id}
    }
}

# Verify every committed row before retrieval so corrupt vectors cannot vanish
# from ranking through SQL NULL. This is a linear pass over the private index.
proc ::rag::sqlite::verify_complete {id dimension} {
    set expected [db onecolumn {SELECT expected FROM corpora WHERE id=$id AND status='complete'}]
    if {$expected eq ""} {error "corpus incomplete or missing"}
    set index 0
    db eval {SELECT sequence,line_data,line_sha,vector,vector_sha FROM fragments WHERE corpus=$id ORDER BY sequence} row {
        if {$row(sequence) != $index || [string length $row(vector)] != 4*$dimension ||
            [digest $row(line_data)] ne $row(line_sha) || [digest $row(vector)] ne $row(vector_sha)} {error "stored corpus integrity error"}
        incr index
    }
    if {$index != $expected} {error "stored corpus incomplete"}
}

# Return ordered {sequence record score} rows from a complete corpus using the
# loaded cosine_similarity_v2 extension; rejects invalid query dimensions.
proc ::rag::sqlite::search {id vector topk} {
    if {![string is integer -strict $topk] || $topk < 1} {error "invalid topk"}
    set dimension [db onecolumn {SELECT dimension FROM corpora WHERE id=$id AND status='complete'}]
    if {$dimension eq ""} {error "corpus incomplete or missing"}
    validate_vector $vector $dimension
    verify_complete $id $dimension
    set result {}
    db eval {SELECT sequence,line_data,line_sha,vector,vector_sha,cosine_similarity_v2(vector,$vector) AS score FROM fragments WHERE corpus=$id AND cosine_similarity_v2(vector,$vector) IS NOT NULL ORDER BY score DESC,sequence ASC LIMIT $topk} row {
        if {[digest $row(line_data)] ne $row(line_sha) || [digest $row(vector)] ne $row(vector_sha)} {error "stored row mismatch"}
        lappend result [list $row(sequence) $row(line_data) $row(score)]
    }
    return $result
}

# Verify the exact immutable embedding configuration before query generation.
proc ::rag::sqlite::check_config {id expected} {
    set hash [digest [encoding convertto utf-8 $expected]]
    set n [db onecolumn {SELECT count(*) FROM corpora WHERE id=$id AND status='complete' AND config=$expected AND config_sha=$hash}]
    if {$n != 1} {error "corpus configuration mismatch"}
}
