# Application Schema PRD: Pharmacy RAG Example

## Purpose
Local-first RAG playground for pharmacy and medical document retrieval, designed to keep clinical documents and model execution inside the user's environment.

## Runtime / Surfaces
- Runtime: Python CLI scripts.
- Ingestion: PDF/DOCX/TXT loaders, LangChain splitters, sentence-transformer embeddings.
- Retrieval: ChromaDB local vector store.
- Generation: Ollama local LLM runtime.
- QA: benchmark/test script for accuracy, safety, and performance checks.

## Core Schema
- `LocalDocument`: source path, title, file type, optional specialty/category.
- `DocumentChunk`: source id, chunk index, text, page number, metadata.
- `EmbeddingRecord`: chunk id, embedding vector, model name, collection id.
- `Query`: question, topK, selected model, output format.
- `RetrievedSource`: chunk id, title, relevance score, L2 distance, location, text sample.
- `RAGResponse`: answer text, sources, model, latency.
- `BenchmarkCase`: test prompt, expected keywords/safety criteria, category.
- `BenchmarkResult`: score, pass/fail, timing, notes.

## Data Stores & Integrations
- `data/` holds user medical documents locally.
- `chroma/` holds generated vector database locally.
- Ollama serves local LLMs; no external API calls are required.
- LangChain coordinates loading, splitting, embedding, retrieval, and answer generation.

## Future Edit Map
- Keep PHI-bearing files out of git and maintain local-only defaults.
- Add a manifest file for document metadata instead of inferring category from folders.
- Version the vector store when chunking, embedding model, or document corpus changes.
- Expand tests before clinical or institutional use.
