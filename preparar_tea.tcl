# preparar_tea.tcl
# Script para estructurar el proyecto Tclllama bajo el estándar TEA
# Autor: El Arquitecto de Yaxché

puts "🏗️  Iniciando andamiaje TEA..."

# 1. Definir la estructura de carpetas estándar
set dirs {
    generic
    library
    tclconfig
    doc
    tests
}

# 2. Crear directorios
foreach d $dirs {
    if {![file exists $d]} {
        puts "   + Creando directorio: $d"
        file mkdir $d
    }
}

# 3. Mover los archivos de C/C++ a 'generic'
# (Aquí vive la lógica dura: tclllama.c, sha256.c)
foreach file [glob -nocomplain *.c *.h *.cpp] {
    puts "   -> Moviendo $file a generic/"
    file rename $file [file join generic $file]
}

# 4. Mover los scripts Tcl a 'library'
# (Aquí vive la lógica de negocio: ollama_registry.tcl)
# Excluimos este mismo script para no moverlo
set my_name [file tail [info script]]
foreach file [glob -nocomplain *.tcl] {
    if {$file ne $my_name && $file ne "pkgIndex.tcl"} {
        puts "   -> Moviendo $file a library/"
        file rename $file [file join library $file]
    }
}

# 5. Crear archivos placeholder para el Cónclave
# (Estos son los que mañana llenaremos con ayuda de la IA)
set placeholders {README.md LICENSE configure.ac Makefile.in pkgIndex.tcl.in}
foreach f $placeholders {
    if {![file exists $f]} {
        set fd [open $f w]
        puts $fd "# Placeholder para $f"
        close $fd
        puts "   + Creado placeholder: $f"
    }
}

puts "✅ Estructura TEA lista."
puts "   Ahora sí, mañana dale 'configure.ac' a claude-code para que lo llene."
