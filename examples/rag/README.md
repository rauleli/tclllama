# Experimental Tcl RAG example

These components adapt persistence and similarity code from `tclembedding`;
see `similarity/LICENSE.tclembedding`. They remain experimental and require
human review of retrieval and answer quality. Keep model weights, databases,
credentials and personal records outside the public checkout.

## Requirements and build

Tcl 8.6, tclllama 0.1.0, Tcllib (`sha256`, `json::write`, `http`), `rl_json`,
SQLite with Tcl bindings, and/or MySQL 8 with `tdbc::mysql` are required. MySQL
must use a private database, InnoDB and a registered `cosine_similarity_v2` UDF.

Build the optional x86-64 SQL extensions into a private directory:

```sh
make -C examples/rag/similarity BUILD_DIR=/private/rag-build
```

The scalar dispatcher is compiled for the x86-64 baseline. Separate SSE2, AVX
and AVX2 objects are selected only after CPU and OS checks. The build requires
SQLite and MySQL development headers; set `MYSQL_CPPFLAGS` if `mysql_config`
is not on PATH. Register the MySQL UDF only in a separately configured private
server; this example does not modify the system plugin directory.

## Configuration and ingestion

The config file is a Tcl dictionary used as data, not executable Tcl. It must
include `model_sha256`, `model_reference`, `model_revision`, `quantization`,
`pooling` (`model`), `normalization` (`l2`), `input_sha256`, `dimension` (384
or 768), `embedding_context`, `document_prefix`, `query_prefix`, `token_budget`,
`batch_size`, optional `threads`, `min_score`, `topk`, `max_context_chars`, `max_history_chars`,
and `generation_max_tokens`. Supply actual verified SHA-256 values and settings
for the selected model. The indexer rechecks file hashes and dimension; model
revision and quantization are caller-supplied metadata, not independently
verified by the program. Prefixes must match the model. The context must not
exceed the model's training context; the binding enforces token and batch limits.

Given a version-1 extractor JSONL and private paths, use:

```sh
tclsh examples/rag/index.tcl sqlite /private/index.sqlite - - create corpus1 source1 /private/input.jsonl /private/config.tcl /private/embed.gguf
tclsh examples/rag/index.tcl sqlite /private/index.sqlite - - resume corpus1 source1 /private/input.jsonl /private/config.tcl /private/embed.gguf
```

To avoid recomputing embeddings and preserve byte-identical vectors, copy a
complete SQLite corpus into an empty dedicated MySQL database:

```sh
tclsh examples/rag/mirror_mysql.tcl /private/index.sqlite /private/mysql.sock database user create corpus1
```

Use `resume` after an interrupted copy. The mirror checks source and destination
hashes before reporting completion. Alternatively, for direct MySQL ingestion
use `mysql /private/mysql.sock database user` in place of
`sqlite /private/index.sqlite - -`. The database must already exist; `create`
creates only the RAG tables in an otherwise empty dedicated database. A password
may be provided through `TCLLLAMA_RAG_MYSQL_PASSWORD`. The command reads and
fragments the whole input in memory before it begins the SQL transaction batches;
large-corpus memory and interruption behavior remain to be measured. A failed
batch does not mark the corpus complete. `resume` checks prior records and the
configuration, but changes in source or configuration require a new corpus.

## Interactive programs

Both clients use identical tclllama query embeddings and SQL retrieval. Their
arguments are backend, location, database, user, SQLite extension, corpus,
embedding GGUF, config file and generator. Use `-` for unused database/user or
SQLite extension arguments. For example:

```sh
tclsh examples/rag/interactive_native.tcl sqlite /private/index.sqlite - - /private/rag-build/sqlite_similarity.so corpus1 /private/embed.gguf /private/config.tcl /private/chat.gguf
tclsh examples/rag/interactive_ollama.tcl sqlite /private/index.sqlite - - /private/rag-build/sqlite_similarity.so corpus1 /private/embed.gguf /private/config.tcl local-model
```

The Ollama client contacts only `127.0.0.1` and uses `/api/chat`; set
`TCLLLAMA_OLLAMA_PORT` to override port 11434. Set `TCLLLAMA_RAG_MIN_SCORE`
and `TCLLLAMA_RAG_TOPK` to adjust retrieval at runtime without rebuilding the
index. Both
provide `/help`, `/mode independent`, `/mode conversation`, `/reset`, `/sources`,
`/exit`, `/quit` and `/salir`. Sources are shown only after `/sources`; answers
receive evidence labels such as `[S1]` for citations. Startup filters native
llama.cpp logs to warnings/errors. Append `--verbose` or `--debug` to either
interactive script for more diagnostics. Context and history limits are in characters; they do not yet
prove that the formatted prompt fits every model's token window. Query thresholds
are model-dependent and require calibration. Review retrieval and response
quality before relying on an answer. Ollama requests `think:false`; actual
model templates and sampling details can differ from the native runtime.

## Tests

Before each retrieval the store checks every committed record and vector hash;
this adds a full linear pass and assumes the database is not concurrently
modified by another application. The `similarity/tests` directory contains synthetic tests. SQLite tests require
the extension path. MySQL tests require a dedicated empty database on a private
TCP-disabled instance with the UDF registered. Never point them at a production
database. Test arguments are documented at the start of each test.
