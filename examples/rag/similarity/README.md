# Similarity component for SQL RAG (work in progress)

These sources and synthetic tests originate from tclembedding commit
`4c73ef2b302b50c5d36902983a2987f19aa303ea`; see `LICENSE.tclembedding`.
They are independent of the main Tcl binding. The kernel accepts IEEE-754
float32 little-endian blobs, rejects malformed or non-finite data and computes
cosine in double precision. The dispatcher can select SSE2, AVX or AVX2 only
after checking CPU and OS support. Build each object with its intended ISA;
never compile the dispatcher or scalar objects with native CPU flags.

This is an incomplete port: build integration, SQL adapters, model ingestion
and interactive examples require further work and validation. Do not treat the
presence of these sources as a validated RAG implementation.
