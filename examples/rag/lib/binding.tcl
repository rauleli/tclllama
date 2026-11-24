# binding.tcl -- Verify the tclllama module used by an experimental RAG index.
# Provides ::rag::binding::verify. Requires Tcl 8.6, sha256 and tclllama loaded.
package require Tcl 8.6
package require sha256
namespace eval ::rag::binding {}

# Verify the loaded Tcl extension against binding_sha256 when it is supplied
# in the config dictionary. Returns its SHA-256; errors if absent or mismatched.
# The module hash is part of the caller's persistent configuration so a
# different build cannot silently resume the same corpus.
proc ::rag::binding::verify {config} {
    set found {}
    foreach item [info loaded] {
        lassign $item path name
        if {[string equal -nocase $name Tclllama]} {lappend found $path}
    }
    if {[llength $found] != 1 || ![file isfile [lindex $found 0]]} {error "cannot identify loaded tclllama module"}
    set hash [::sha2::sha256 -hex -filename [lindex $found 0]]
    if {[dict exists $config binding_sha256] && [dict get $config binding_sha256] ne $hash} {error "loaded binding hash mismatch"}
    return $hash
}
