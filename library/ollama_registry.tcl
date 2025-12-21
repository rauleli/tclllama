# ollama_registry.tcl - Cliente de Registry de Ollama v2.1
# MEJORA: Timeout por chunk (30s sin datos) en lugar de timeout global
# Uso: source ollama_registry.tcl

package require http

# Cargar binding SHA256 rápido (OpenSSL) si está disponible
set _sha256_fast 0
if {[file exists "[file dirname [info script]]/sha256.so"]} {
    if {[catch {load "[file dirname [info script]]/sha256.so" Sha256}]} {
        package require sha256
    } else {
        set _sha256_fast 1
    }
} else {
    package require sha256
}

# Habilitar HTTPS con TLS
if {[catch {package require tls}]} {
    puts "❌ Error: El paquete 'tls' no está disponible"
    puts "   Instala tcl-tls:"
    puts "   • Debian/Ubuntu: sudo apt-get install tcl-tls"
    puts "   • Fedora/RHEL: sudo dnf install tcl-tls"
    puts "   • macOS: brew install tcl-tk"
    error "TLS package required for HTTPS"
}

::http::register https 443 [list ::tls::socket -autoservername true]

namespace eval ollama_registry {
    variable registry_host "registry.ollama.ai"
    variable registry_port 443
    variable registry_scheme "https"
    variable models_dir "./modelos"
    variable verbose 1
    variable catalog_cache {}
    variable use_fast_sha256 $::_sha256_fast
    
    # Mejoras v2.0: configuración de reintentos
    variable max_retries 3
    variable retry_delay 2000
    variable request_timeout 30000
    
    # ✅ MEJORA v2.1: TIMEOUT POR CHUNK, NO GLOBAL
    variable chunk_timeout 30000     ;# 30 segundos máximo SIN DATOS por chunk
    variable chunk_blocksize 65536   ;# Tamaño de lectura (igual que -blocksize)
    
    # Thread-safety: estados de descarga individuales
    variable download_states [dict create]
    variable state_counter 0
}

# ============================================
# GESTIÓN DE ESTADOS DE DESCARGA (Thread-safe)
# ============================================

proc ollama_registry::create_download_state {output_path} {
    variable download_states
    variable state_counter
    variable use_fast_sha256
    
    incr state_counter
    set state_id "dl_$state_counter"
    
    dict set download_states $state_id [dict create \
        fd "" \
        sha_ctx "" \
        bytes 0 \
        output_path $output_path \
        fast_sha $use_fast_sha256 \
        last_progress -1]
    
    return $state_id
}

proc ollama_registry::get_download_state {state_id key} {
    variable download_states
    if {![dict exists $download_states $state_id]} {
        error "Invalid state_id: $state_id"
    }
    return [dict get $download_states $state_id $key]
}

proc ollama_registry::set_download_state {state_id key value} {
    variable download_states
    if {![dict exists $download_states $state_id]} {
        error "Invalid state_id: $state_id"
    }
    dict set download_states $state_id $key $value
}

proc ollama_registry::cleanup_download_state {state_id} {
    variable download_states
    set fd [dict get $download_states $state_id fd]
    catch {close $fd}
    dict unset download_states $state_id
}

# ============================================
# CALLBACK DE PROGRESO
# ============================================

proc ollama_registry::progress_callback {state_id token totalsize currentsize} {
    if {![string is integer -strict $currentsize]} { return }
    
    set downloaded_mb [format "%.1f" [expr {$currentsize / 1048576.0}]]

    if {$totalsize > 0} {
        set progress [expr {($currentsize * 100.0) / $totalsize}]
        set last_progress [get_download_state $state_id last_progress]
        
        if {int($progress) > $last_progress} {
            set_download_state $state_id last_progress [expr {int($progress)}]
            set total_mb [format "%.1f" [expr {$totalsize / 1048576.0}]]
            puts -nonewline "\r   Progreso: ${downloaded_mb}/${total_mb}MB ([format "%.0f" $progress]%)"
            flush stdout
        }
    } else {
        # PLAN B: Si no sabemos el total, solo mostrar lo descargado
        # Actualizamos cada 1MB para no saturar la consola
        set last_mb [get_download_state $state_id last_progress] 
        set current_mb_int [expr {int($currentsize / 1048576)}]
        
        if {$current_mb_int > $last_mb} {
            set_download_state $state_id last_progress $current_mb_int
            puts -nonewline "\r   Descargado: ${downloaded_mb} MB (Total desconocido)"
            flush stdout
        }
    }
}

# ============================================
# UTILIDADES DE SISTEMA DE ARCHIVOS
# ============================================

proc ollama_registry::ensure_dir {path} {
    if {![file exists $path]} {
        file mkdir $path
    }
}

proc ollama_registry::init_storage {} {
    variable models_dir

    ensure_dir $models_dir
    ensure_dir "$models_dir/manifests"
    ensure_dir "$models_dir/manifests/library"
    ensure_dir "$models_dir/blobs"

    set readme_path "$models_dir/README.md"
    if {![file exists $readme_path]} {
        set fd [open $readme_path w]
        puts $fd "# Ollama Registry Local Storage"
        puts $fd ""
        puts $fd "Este directorio almacena modelos descargados desde registry.ollama.ai"
        puts $fd ""
        puts $fd "## Estructura:"
        puts $fd ""
        puts $fd "```"
        puts $fd "modelos/"
        puts $fd "  ├── manifests/library/{modelo}/{tag}  - Manifests JSON"
        puts $fd "  └── blobs/sha256-{digest}             - Blobs de datos"
        puts $fd "```"
        puts $fd ""
        puts $fd "## Tipos de Blobs:"
        puts $fd ""
        puts $fd "- **application/vnd.ollama.image.model** - Archivo GGUF del modelo"
        puts $fd "- **application/vnd.ollama.image.template** - Template de chat"
        puts $fd "- **application/vnd.ollama.image.system** - System prompt"
        puts $fd "- **application/vnd.ollama.image.params** - Parámetros del modelo"
        puts $fd "- **application/vnd.docker.container.image.v1+json** - Config del modelo"
        puts $fd ""
        puts $fd "## Verificación:"
        puts $fd ""
        puts fd "Todos los blobs son verificados con SHA256 después de la descarga."
        puts $fd ""
        close $fd
    }
}

# ============================================
# UTILIDADES HTTP CON REINTENTOS
# ============================================

proc ollama_registry::http_get {path {headers {}}} {
    variable registry_scheme
    variable registry_host
    variable max_retries
    variable retry_delay
    variable request_timeout

    set url "${registry_scheme}://${registry_host}${path}"
    set header_list [list "Accept" "application/vnd.docker.distribution.manifest.v2+json"]
    foreach {key value} $headers {
        lappend header_list $key $value
    }

    set last_error ""
    for {set attempt 0} {$attempt <= $max_retries} {incr attempt} {
        
        set status_code [catch {
            set token [::http::geturl $url \
                -timeout $request_timeout \
                -headers $header_list]
            
            set status [::http::status $token]
            set ncode [::http::ncode $token]
            set data [::http::data $token]
            
            ::http::cleanup $token
            
            if {$status ne "ok"} {
                error "HTTP error: $status"
            }
            
            if {$ncode != 200} {
                error "HTTP $ncode: $data"
            }
            
            set data
        } result]

        if {$status_code == 0 || $status_code == 2} {
            return $result
        } else {
            set last_error $result
            if {$attempt < $max_retries} {
                after $retry_delay
            }
        }
    }
    
    return -code error "Failed after $max_retries retries: $last_error"
}

proc ollama_registry::http_download_file {path output_path {progress_callback {}}} {
    variable registry_scheme
    variable registry_host
    variable max_retries
    variable retry_delay

    set url "${registry_scheme}://${registry_host}${path}"
    
    set last_error ""
    for {set attempt 0} {$attempt <= $max_retries} {incr attempt} {
        set state_id [create_download_state $output_path]
        
        set status_code [catch {
            set result [download_file_attempt $url $state_id $progress_callback]
            cleanup_download_state $state_id
            set result
        } result]
        
        if {$status_code == 0 || $status_code == 2} {
            return $result
        }
        
        cleanup_download_state $state_id
        catch {file delete $output_path}
        
        set last_error $result
        if {$attempt < $max_retries} {
            after $retry_delay
        }
    }
    
    return -code error "Download failed after $max_retries retries: $last_error"
}

proc ollama_registry::download_file_attempt {url state_id progress_callback} {
    variable use_fast_sha256

    set output_path [get_download_state $state_id output_path]
    set max_redirects 5
    set redirect_count 0
    set current_url $url

    while {$redirect_count < $max_redirects} {
        set fd [open $output_path w]
        fconfigure $fd -translation binary -buffering full
        set_download_state $state_id fd $fd
        
        if {$use_fast_sha256} {
            set_download_state $state_id sha_ctx [sha256::init]
        } else {
            set_download_state $state_id sha_ctx [::sha2::SHA256Init]
        }
        set_download_state $state_id bytes 0

        set handler_cmd [list ::ollama_registry::download_handler $state_id $progress_callback]
        
        # ✅ v2.1: SIN TIMEOUT GLOBAL, se maneja por chunk en el handler
        set token [::http::geturl $current_url \
            -binary 1 \
            -blocksize 65536 \
            -handler $handler_cmd]

        set status [::http::status $token]
        set ncode [::http::ncode $token]
        set meta [::http::meta $token]

        catch {close [get_download_state $state_id fd]}
        set_download_state $state_id fd ""

        if {$status eq "ok" && $ncode == 200} {
            set sha_ctx [get_download_state $state_id sha_ctx]
            if {$use_fast_sha256} {
                set hex_hash [sha256::final $sha_ctx]
            } else {
                set hash_bytes [::sha2::SHA256Final $sha_ctx]
                set hex_hash [binary encode hex $hash_bytes]
            }
            set file_size [get_download_state $state_id bytes]

            ::http::cleanup $token
            return [list $file_size $hex_hash]
        }

        if {$status eq "ok" && ($ncode == 301 || $ncode == 302 || $ncode == 307 || $ncode == 308)} {
            ::http::cleanup $token

            set location ""
            dict for {key value} $meta {
                if {[string tolower $key] eq "location"} {
                    set location $value
                    break
                }
            }

            if {$location eq ""} {
                error "Redirect without Location header (code: $ncode)"
            }

            set current_url $location
            incr redirect_count
            continue
        }

        ::http::cleanup $token
        error "Download failed: $status (code: $ncode)"
    }

    error "Too many redirects (>$max_redirects)"
}

# ============================================
# HANDLER MEJORADO CON TIMEOUT POR CHUNK (v2.1)
# ============================================

proc ollama_registry::download_handler {state_id progress_callback socket token} {
    variable chunk_timeout
    variable chunk_blocksize
    
    # 1. Configurar socket para lectura no-bloqueante
    fconfigure $socket -blocking 0 -buffering full -translation binary
    
    set last_data_time [clock milliseconds]
    set total_bytes_read 0
    
    while {![eof $socket]} {
        # 2. Intentar leer un chunk
        set data [read $socket $chunk_blocksize]
        set bytes_read [string length $data]
        
        if {$bytes_read > 0} {
            # ✅ DATOS RECIBIDOS: procesar normalmente
            set total_bytes_read [expr {$total_bytes_read + $bytes_read}]
            set last_data_time [clock milliseconds]  # Reset timer
            
            # Escribir al archivo
            set fd [get_download_state $state_id fd]
            puts -nonewline $fd $data

            # Actualizar SHA256
            set sha_ctx [get_download_state $state_id sha_ctx]
            set fast_sha [get_download_state $state_id fast_sha]
            if {$fast_sha} {
                sha256::update $sha_ctx $data
            } else {
                ::sha2::SHA256Update $sha_ctx $data
            }

            # Actualizar contador de bytes
            set bytes [get_download_state $state_id bytes]
            set bytes [expr {$bytes + $bytes_read}]
            set_download_state $state_id bytes $bytes

            # Mostrar progreso si hay callback
            if {$progress_callback ne ""} {
                upvar #0 $token state
                set total 0
                if {[info exists state(totalsize)]} {
                    set total $state(totalsize)
                }
                {*}$progress_callback $state_id $token $total $bytes
            }
            
        } else {
            # ⏰ VERIFICAR TIMEOUT: ¿demasiado tiempo sin datos?
            set idle_time [expr {[clock milliseconds] - $last_data_time}]
            if {$idle_time > $chunk_timeout} {
                return -code error "Network timeout: no data received in [expr {$chunk_timeout/1000}] seconds"
            }
            
            # Pequeña pausa para no consumir CPU
            after 100
        }
    }
    
    return $total_bytes_read
}

# ============================================
# UTILIDADES DE CHECKSUM
# ============================================

proc ollama_registry::sha256_file {filepath} {
    variable use_fast_sha256

    if {$use_fast_sha256} {
        return [sha256::file $filepath]
    }

    set fd [open $filepath r]
    fconfigure $fd -translation binary

    set ctx [::sha2::SHA256Init]

    while {![eof $fd]} {
        set chunk [read $fd 65536]
        ::sha2::SHA256Update $ctx $chunk
    }

    close $fd

    set hash [::sha2::SHA256Final $ctx]
    return [binary encode hex $hash]
}

proc ollama_registry::verify_blob {filepath expected_digest} {
    if {![regexp {^sha256:(.+)$} $expected_digest -> expected_hash]} {
        return -code error "Invalid digest format: $expected_digest"
    }

    set actual_hash [sha256_file $filepath]

    if {$actual_hash ne $expected_hash} {
        return -code error "Checksum mismatch: expected $expected_hash, got $actual_hash"
    }

    return 1
}

# ============================================
# JSON PARSING
# ============================================

proc ollama_registry::parse_json {json_str} {
    if {[catch {package require json}]} {
        if {[catch {package require json::write}]} {
            return -code error "JSON package not available. Install tcllib: sudo apt-get install tcllib"
        }
    }

    return [::json::json2dict $json_str]
}

# ============================================
# REGISTRY API - MANIFESTS
# ============================================

proc ollama_registry::fetch_manifest {model tag} {
    variable verbose
    variable models_dir

    if {$verbose} {
        puts "📥 Descargando manifest: $model:$tag"
    }

    set api_path "/v2/library/$model/manifests/$tag"
    set manifest_json [http_get $api_path]
    set manifest [parse_json $manifest_json]

    set manifest_dir "$models_dir/manifests/library/$model"
    ensure_dir $manifest_dir

    set manifest_file "$manifest_dir/$tag"
    set fd [open $manifest_file w]
    puts $fd $manifest_json
    close $fd

    if {$verbose} {
        puts "✅ Manifest guardado: $manifest_file"
    }

    return $manifest
}

# ============================================
# REGISTRY API - BLOBS
# ============================================

proc ollama_registry::download_blob {model digest expected_size {progress_callback {}}} {
    variable verbose
    variable models_dir

    set blob_filename [string map {: -} $digest]
    set blob_path "$models_dir/blobs/$blob_filename"

    if {[file exists $blob_path]} {
        set actual_size [file size $blob_path]

        if {$actual_size == $expected_size} {
            if {$verbose} {
                puts "✓ Blob ya existe: $blob_filename (tamaño verificado)"
            }
            return $blob_path
        }

        if {$verbose} {
            puts "⚠️  Tamaño incorrecto (esperado: $expected_size, actual: $actual_size), redescargando..."
        }
        file delete $blob_path
    }

    if {$verbose} {
        set size_mb [format "%.1f" [expr {$expected_size / 1048576.0}]]
        puts "📥 Descargando blob: $blob_filename (${size_mb}MB)"
    }

    set api_path "/v2/library/$model/blobs/$digest"

    set formatted_callback {}
    if {$verbose} {
        set formatted_callback ::ollama_registry::progress_callback
    } elseif {$progress_callback ne ""} {
        set formatted_callback $progress_callback
    }

    if {[catch {
        set result [http_download_file $api_path $blob_path $formatted_callback]
        lassign $result downloaded_bytes calculated_hash
    } error]} {
        if {$verbose} {
            puts ""
            puts "❌ Error descargando blob: $error"
        }
        return -code error $error
    }

    if {$verbose} {
        puts ""
        puts "🔍 Verificando SHA256..."
    }

    if {![regexp {^sha256:(.+)$} $digest -> expected_hash]} {
        file delete $blob_path
        return -code error "Formato de digest inválido: $digest"
    }

    if {$calculated_hash ne $expected_hash} {
        file delete $blob_path
        if {$verbose} {
            puts "❌ Checksum mismatch!"
            puts "   Esperado: $expected_hash"
            puts "   Obtenido: $calculated_hash"
        }
        return -code error "Checksum mismatch: esperado $expected_hash, obtenido $calculated_hash"
    }

    if {$verbose} {
        puts "✅ Blob descargado y verificado"
    }

    return $blob_path
}

# ============================================
# FUNCIONES DE ALTO NIVEL
# ============================================

proc ollama_registry::download_model {model_name args} {
    variable verbose
    variable models_dir

    init_storage

    set parts [split $model_name ":"]
    if {[llength $parts] == 2} {
        set model [lindex $parts 0]
        set tag [lindex $parts 1]
    } else {
        set model $model_name
        set tag "latest"
    }

    if {$verbose} {
        puts "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        puts "🚀 Descargando modelo: $model:$tag"
        puts "   Registry: registry.ollama.ai"
        puts "   Destino: $models_dir"
        puts "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        puts ""
    }

    set start_time [clock milliseconds]

    if {[catch {
        set manifest [fetch_manifest $model $tag]
    } error]} {
        return -code error "Error obteniendo manifest: $error"
    }

    puts ""

    if {[dict exists $manifest config]} {
        set config [dict get $manifest config]
        set config_digest [dict get $config digest]
        set config_size [dict get $config size]

        if {$verbose} {
            puts "📄 Descargando configuración..."
        }

        download_blob $model $config_digest $config_size
        puts ""
    }

    if {[dict exists $manifest layers]} {
        set layers [dict get $manifest layers]
        set total_layers [llength $layers]
        set current 1

        foreach layer $layers {
            set layer_digest [dict get $layer digest]
            set layer_size [dict get $layer size]
            set layer_type [dict get $layer mediaType]

            if {$verbose} {
                puts "📦 Layer $current/$total_layers - $layer_type"
            }

            download_blob $model $layer_digest $layer_size
            puts ""

            incr current
        }
    }

    set elapsed [expr {([clock milliseconds] - $start_time) / 1000.0}]

    if {$verbose} {
        puts "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        puts "✅ Modelo descargado exitosamente en [format "%.1f" $elapsed]s"
        puts "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        puts ""
    }

    return [list $model $tag]
}

proc ollama_registry::list_local_models {} {
    variable models_dir

    init_storage

    set manifest_dir "$models_dir/manifests/library"

    if {![file exists $manifest_dir]} {
        return {}
    }

    set models {}

    foreach model_dir [glob -nocomplain -directory $manifest_dir -type d *] {
        set model_name [file tail $model_dir]

        foreach tag_file [glob -nocomplain -directory $model_dir *] {
            if {[file isfile $tag_file]} {
                set tag [file tail $tag_file]
                lappend models "$model_name:$tag"
            }
        }
    }

    return [lsort $models]
}

proc ollama_registry::delete_model {model_name} {
    variable verbose
    variable models_dir

    set parts [split $model_name ":"]
    if {[llength $parts] == 2} {
        set model [lindex $parts 0]
        set tag [lindex $parts 1]
    } else {
        set model $model_name
        set tag "latest"
    }

    set manifest_file "$models_dir/manifests/library/$model/$tag"

    if {![file exists $manifest_file]} {
        return -code error "Modelo no encontrado: $model:$tag"
    }

    if {$verbose} {
        puts "🗑️  Eliminando modelo: $model:$tag"
    }

    set fd [open $manifest_file r]
    set manifest_json [read $fd]
    close $fd

    set manifest [parse_json $manifest_json]
    set blobs_to_check {}

    if {[dict exists $manifest config]} {
        lappend blobs_to_check [dict get $manifest config digest]
    }

    if {[dict exists $manifest layers]} {
        foreach layer [dict get $manifest layers] {
            lappend blobs_to_check [dict get $layer digest]
        }
    }

    file delete $manifest_file

    set model_dir "$models_dir/manifests/library/$model"
    if {[llength [glob -nocomplain -directory $model_dir *]] == 0} {
        file delete $model_dir
    }

    set other_models [list_local_models]
    set used_blobs {}

    foreach other_model $other_models {
        set parts [split $other_model ":"]
        set other_name [lindex $parts 0]
        set other_tag [lindex $parts 1]

        set other_manifest_file "$models_dir/manifests/library/$other_name/$other_tag"

        if {[file exists $other_manifest_file]} {
            set fd [open $other_manifest_file r]
            set other_json [read $fd]
            close $fd

            set other_manifest [parse_json $other_json]

            if {[dict exists $other_manifest config]} {
                lappend used_blobs [dict get $other_manifest config digest]
            }

            if {[dict exists $other_manifest layers]} {
                foreach layer [dict get $other_manifest layers] {
                    lappend used_blobs [dict get $layer digest]
                }
            }
        }
    }

    set deleted_count 0
    foreach blob_digest $blobs_to_check {
        if {$blob_digest ni $used_blobs} {
            set blob_filename [string map {: -} $blob_digest]
            set blob_path "$models_dir/blobs/$blob_filename"

            if {[file exists $blob_path]} {
                file delete $blob_path
                incr deleted_count
            }
        }
    }

    if {$verbose} {
        puts "✅ Modelo eliminado ($deleted_count blobs eliminados)"
    }

    return 1
}

proc ollama_registry::get_model_path {model_name} {
    variable models_dir

    set parts [split $model_name ":"]
    if {[llength $parts] == 2} {
        set model [lindex $parts 0]
        set tag [lindex $parts 1]
    } else {
        set model $model_name
        set tag "latest"
    }

    set manifest_file "$models_dir/manifests/library/$model/$tag"

    if {![file exists $manifest_file]} {
        return -code error "Modelo no encontrado: $model:$tag"
    }

    set fd [open $manifest_file r]
    set manifest_json [read $fd]
    close $fd

    set manifest [parse_json $manifest_json]

    if {[dict exists $manifest layers]} {
        foreach layer [dict get $manifest layers] {
            set media_type [dict get $layer mediaType]

            if {$media_type eq "application/vnd.ollama.image.model"} {
                set digest [dict get $layer digest]
                set blob_filename [string map {: -} $digest]
                set blob_path "$models_dir/blobs/$blob_filename"

                if {[file exists $blob_path]} {
                    return $blob_path
                }
            }
        }
    }

    return -code error "Archivo del modelo no encontrado en los blobs"
}

# ============================================
# FUNCIÓN NUEVA: show_config (v2.1)
# ============================================

proc ollama_registry::show_config {} {
    variable registry_host
    variable registry_scheme
    variable models_dir
    variable max_retries
    variable retry_delay
    variable request_timeout
    variable chunk_timeout
    variable chunk_blocksize
    variable use_fast_sha256
    
    puts "⚙️  Configuración Ollama Registry v2.1:"
    puts "   Registry: ${registry_scheme}://${registry_host}"
    puts "   Directorio: ${models_dir}"
    puts "   Max reintentos: ${max_retries}"
    puts "   Delay reintentos: [expr {$retry_delay / 1000}]s"
    puts "   Timeout peticiones: [expr {$request_timeout / 1000}]s"
    puts "   ✅ Timeout por chunk: [expr {$chunk_timeout / 1000}]s (v2.1)"
    puts "   Tamaño chunk: [expr {$chunk_blocksize / 1024}]KB"
    puts "   SHA256 rápido: ${use_fast_sha256}"
}

# ============================================
# UTILIDADES
# ============================================

proc ollama_registry::silent {{mode 1}} {
    variable verbose
    set verbose [expr {!$mode}]
}

proc ollama_registry::show_model_info {model_name} {
    variable models_dir

    set parts [split $model_name ":"]
    if {[llength $parts] == 2} {
        set model [lindex $parts 0]
        set tag [lindex $parts 1]
    } else {
        set model $model_name
        set tag "latest"
    }

    set manifest_file "$models_dir/manifests/library/$model/$tag"

    if {![file exists $manifest_file]} {
        return -code error "Modelo no encontrado: $model:$tag"
    }

    set fd [open $manifest_file r]
    set manifest_json [read $fd]
    close $fd

    set manifest [parse_json $manifest_json]

    puts "📊 Información del modelo: $model:$tag"
    puts ""
    puts "Schema Version: [dict get $manifest schemaVersion]"
    puts "Media Type: [dict get $manifest mediaType]"
    puts ""

    if {[dict exists $manifest config]} {
        set config [dict get $manifest config]
        puts "Config:"
        puts "  Size: [format "%.1f" [expr {[dict get $config size] / 1024.0}]]KB"
        puts "  Digest: [dict get $config digest]"
        puts ""
    }

    if {[dict exists $manifest layers]} {
        puts "Layers:"
        set total_size 0
        foreach layer [dict get $manifest layers] {
            set layer_size [dict get $layer size]
            set layer_type [dict get $layer mediaType]
            set layer_digest [dict get $layer digest]

            set size_str [format "%.1f" [expr {$layer_size / 1048576.0}]]
            puts "  • $layer_type"
            puts "    Size: ${size_str}MB"
            puts "    Digest: [string range $layer_digest 0 40]..."

            set total_size [expr {$total_size + $layer_size}]
        }

        puts ""
        puts "Total Size: [format "%.2f" [expr {$total_size / 1073741824.0}]]GB"
    }
}

# ============================================
# INFORMACIÓN v2.1
# ============================================

puts "✅ ollama_registry.tcl v2.1 cargado"
puts "   Mejora: Timeout por chunk (30s sin datos), no timeout global"
puts ""
puts "   Comandos principales:"
puts "   • ollama_registry::download_model <modelo:tag>"
puts "   • ollama_registry::list_local_models"
puts "   • ollama_registry::delete_model <modelo:tag>"
puts "   • ollama_registry::get_model_path <modelo:tag>"
puts "   • ollama_registry::show_model_info <modelo:tag>"
puts "   • ollama_registry::show_config"
puts ""
