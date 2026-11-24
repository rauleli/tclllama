# mysql_store.tcl -- MySQL/InnoDB persistence for RAG records and f32le vectors.
# Provides ::rag::mysql; requires Tcl 8.6, tdbc::mysql and sha256.
# Adapted from tclembedding's examples/03-chatgpt-export/mysql_retrieval.tcl;
# see ../similarity/LICENSE.tclembedding. Use a private database and socket.
package require Tcl 8.6
package require tdbc::mysql
package require sha256

# One explicitly selected database connection per namespace.
namespace eval ::rag::mysql {}

# Validate f32le vector shape, finiteness and nonzero norm before SQL use.
proc ::rag::mysql::validate_vector {vector dimension} {
    if {[string length $vector] != 4*$dimension} {error "vector dimension mismatch"}
    binary scan $vector r* values
    set norm 0.0
    foreach value $values {
        if {[catch {expr {double($value)==double($value) && abs(double($value)) < Inf}} finite] || !$finite} {error "non-finite vector"}
        set norm [expr {$norm+double($value)*double($value)}]
    }
    if {$norm == 0.0} {error "zero vector"}
}

# Hash a byte sequence with SHA-256 and return hexadecimal text.
proc ::rag::mysql::digest {bytes} {
    return [::sha2::sha256 -hex $bytes]
}

# Open the supplied local socket and existing database using the given account.
proc ::rag::mysql::open {socket database user} {
    if {![file exists $socket] || $database eq "" || $user eq ""} {error "invalid connection arguments"}
    set options [list -socket $socket -database $database -user $user]
    if {[info exists ::env(TCLLLAMA_RAG_MYSQL_PASSWORD)]} {
        lappend options -password $::env(TCLLLAMA_RAG_MYSQL_PASSWORD)
    }
    tdbc::mysql::connection create ::rag::mysql::db {*}$options
}

# Close the current connection.
proc ::rag::mysql::close {} {
    db close
}

# Execute a prepared statement, binding named binary parameters as varbinary.
# Returns a list of result rows and closes statement and result resources.
proc ::rag::mysql::run {sql values {binary_names {}}} {
    set stmt [db prepare $sql]
    try {
        foreach name $binary_names {$stmt paramtype $name varbinary}
        set rs [$stmt execute $values]
        try {
            set rows {}
            while {[$rs nextlist row]} {lappend rows $row}
            return $rows
        } finally {$rs close}
    } finally {$stmt close}
}

# Initialize tables in an otherwise empty dedicated database. Existing tables
# cause an error so no preexisting data is replaced.
proc ::rag::mysql::init {} {
    set n [lindex [run {SELECT count(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name IN ('rag_corpora','rag_fragments')} {}] 0 0]
    if {$n != 0} {error "store already exists or is incomplete"}
    run {CREATE TABLE rag_corpora (id VARCHAR(191) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY, config LONGBLOB NOT NULL, config_sha CHAR(64) NOT NULL, dimension INT NOT NULL, expected BIGINT NOT NULL, progress BIGINT NOT NULL, status VARCHAR(16) NOT NULL) ENGINE=InnoDB} {}
    run {CREATE TABLE rag_fragments (corpus VARCHAR(191) CHARACTER SET ascii COLLATE ascii_bin NOT NULL, sequence_no BIGINT NOT NULL, line_data LONGBLOB NOT NULL, line_sha CHAR(64) NOT NULL, vector BLOB NOT NULL, vector_sha CHAR(64) NOT NULL, PRIMARY KEY(corpus,sequence_no), FOREIGN KEY(corpus) REFERENCES rag_corpora(id)) ENGINE=InnoDB} {}
}

# Start or resume a corpus and verify its immutable identity and checkpoint.
proc ::rag::mysql::begin {id config dimension expected} {
    if {![string is integer -strict $dimension] || $dimension ni {384 768} || ![string is integer -strict $expected] || $expected < 1} {error "invalid dimension or expected count"}
    set config [encoding convertto utf-8 $config]
    set hash [digest $config]
    db begintransaction
    try {
        set state [run {SELECT config,config_sha,dimension,expected,progress,status FROM rag_corpora WHERE id=:id FOR UPDATE} [list id $id]]
        if {![llength $state]} {
            run {INSERT INTO rag_corpora VALUES(:id,:config,:hash,:dimension,:expected,0,'building')} [list id $id config $config hash $hash dimension $dimension expected $expected] {config}
            set progress 0
        } else {
            lassign [lindex $state 0] saved savedhash d e progress status
            if {$saved ne $config || $savedhash ne $hash || $d != $dimension || $e != $expected || $status ni {building complete}} {error "corpus configuration conflict"}
        }
        lassign [lindex [run {SELECT count(*),max(sequence_no) FROM rag_fragments WHERE corpus=:id} [list id $id]] 0] count last
        if {$count != $progress || ($progress > 0 && $last != $progress-1)} {error "checkpoint mismatch"}
        db commit
    } on error {message options} {
        catch {db rollback}
        return -options $options $message
    }
    return $progress
}

# Atomically append contiguous {sequence utf8_record vector} rows and advance
# the checkpoint. Invalid vectors, failed inserts and changed state roll back.
proc ::rag::mysql::append {id rows} {
    if {![llength $rows]} {error "empty batch"}
    db begintransaction
    try {
        set state [run {SELECT progress,dimension,expected FROM rag_corpora WHERE id=:id AND status='building' FOR UPDATE} [list id $id]]
        if {[llength $state] != 1} {error "corpus is not building"}
        lassign [lindex $state 0] progress dimension expected
        foreach row $rows {
            if {[llength $row] != 3} {error "invalid row"}
            lassign $row sequence line vector
            if {$sequence != $progress || $sequence >= $expected || [string length $vector] != 4*$dimension} {error "invalid sequence or vector dimension"}
            binary scan $vector r* values
            set norm 0.0
            foreach value $values {
                if {[catch {expr {double($value)==double($value) && abs(double($value)) < Inf}} finite] || !$finite} {error "non-finite vector"}
                set norm [expr {$norm+double($value)*double($value)}]
            }
            if {$norm == 0.0} {error "zero vector"}
            set line_sha [digest $line]
            set vector_sha [digest $vector]
            run {INSERT INTO rag_fragments VALUES(:id,:sequence,:line,:line_sha,:vector,:vector_sha)} [list id $id sequence $sequence line $line line_sha $line_sha vector $vector vector_sha $vector_sha] {line vector}
            incr progress
        }
        run {UPDATE rag_corpora SET progress=:progress WHERE id=:id} [list id $id progress $progress]
        db commit
    } on error {message options} {
        catch {db rollback}
        return -options $options $message
    }
    return $progress
}

# Compare a previously committed record and both hashes during resumption.
proc ::rag::mysql::verify {id sequence line} {
    set rows [run {SELECT line_data,line_sha,vector,vector_sha FROM rag_fragments WHERE corpus=:id AND sequence_no=:sequence} [list id $id sequence $sequence]]
    if {[llength $rows] != 1} {error "missing stored row"}
    lassign [lindex $rows 0] saved line_sha vector vector_sha
    if {$saved ne $line || [digest $saved] ne $line_sha || [digest $vector] ne $vector_sha} {error "stored row mismatch"}
}

# Expose a corpus only after expected rows and checkpoint agree.
proc ::rag::mysql::finish {id} {
    db begintransaction
    try {
        set state [run {SELECT expected,progress FROM rag_corpora WHERE id=:id FOR UPDATE} [list id $id]]
        set count [lindex [run {SELECT count(*) FROM rag_fragments WHERE corpus=:id} [list id $id]] 0 0]
        if {[llength $state] != 1 || [lindex $state 0 0] != $count || [lindex $state 0 1] != $count} {error "incomplete corpus"}
        run {UPDATE rag_corpora SET status='complete' WHERE id=:id} [list id $id]
        db commit
    } on error {message options} {
        catch {db rollback}
        return -options $options $message
    }
}

# Verify a completed corpus in MySQL before searching. SHA-256 and shape checks
# run server-side so a large index does not transfer every BLOB to Tcl. The
# primary key makes count/min/max sufficient to prove contiguous sequences.
proc ::rag::mysql::verify_complete {id dimension} {
    set state [run {SELECT expected FROM rag_corpora WHERE id=:id AND status='complete'} [list id $id]]
    if {[llength $state] != 1} {error "corpus incomplete or missing"}
    set expected [lindex $state 0 0]
    set size [expr {4*$dimension}]
    set result [run {SELECT COUNT(*),MIN(sequence_no),MAX(sequence_no),
        SUM(OCTET_LENGTH(vector)<>:size),
        SUM(BINARY line_sha<>BINARY SHA2(line_data,256)),
        SUM(BINARY vector_sha<>BINARY SHA2(vector,256))
        FROM rag_fragments WHERE corpus=:id} [list id $id size $size]]
    if {[llength $result] != 1} {error "stored corpus integrity error"}
    lassign [lindex $result 0] count first last bad_size bad_lines bad_vectors
    if {$count != $expected || $first != 0 || $last != $expected-1 ||
        $bad_size != 0 || $bad_lines != 0 || $bad_vectors != 0} {
        error "stored corpus integrity error"
    }
}

# Retrieve ordered {sequence record score} rows via the registered UDF.
proc ::rag::mysql::search {id vector topk} {
    if {![string is integer -strict $topk] || $topk < 1} {error "invalid topk"}
    set state [run {SELECT dimension FROM rag_corpora WHERE id=:id AND status='complete'} [list id $id]]
    if {[llength $state] != 1} {error "corpus incomplete or missing"}
    set dimension [lindex $state 0 0]
    validate_vector $vector $dimension
    verify_complete $id $dimension
    set rows [run {SELECT sequence_no,line_data,line_sha,vector,vector_sha,cosine_similarity_v2(vector,:vector) AS score FROM rag_fragments WHERE corpus=:id AND cosine_similarity_v2(vector,:vector) IS NOT NULL ORDER BY score DESC,sequence_no ASC LIMIT :topk} [list id $id vector $vector topk $topk] {vector}]
    set result {}
    foreach row $rows {
        lassign $row sequence line line_sha stored_vector vector_sha score
        if {[digest $line] ne $line_sha || [digest $stored_vector] ne $vector_sha} {error "stored row mismatch"}
        lappend result [list $sequence $line $score]
    }
    return $result
}

# Verify the exact immutable embedding configuration before query generation.
proc ::rag::mysql::check_config {id expected} {
    set expected [encoding convertto utf-8 $expected]
    set hash [digest $expected]
    set rows [run {SELECT count(*) FROM rag_corpora WHERE id=:id AND status='complete' AND config=:expected AND config_sha=:hash} [list id $id expected $expected hash $hash] {expected}]
    if {[lindex $rows 0 0] != 1} {error "corpus configuration mismatch"}
}
