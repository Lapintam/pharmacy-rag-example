# Pharmacy RAG System — Product Requirements Document (Scaffold)

> One-page schema reference. Living document; refine as features harden.

## 1. Product

A **fully local** Retrieval-Augmented Generation playground for pharmacy/medical knowledge retrieval. Designed for healthcare environments where data must not leave the machine: documents are ingested, embedded, and queried entirely on-device via Ollama + ChromaDB. Every answer cites source filenames + page numbers; benchmarks emphasize safety-critical recall (medication contraindications, overdoses, pediatric dosing).

## 2. Users & Roles

| Role | Capability |
|---|---|
| **Operator** | Drops documents into `data/`, runs ingestion, queries CLI |
| **Reviewer** | Runs `test_rag_system.py` to validate accuracy + safety |

No multi-user model — single-machine, single-tenant by design.

## 3. Architecture

A small Python CLI built on LangChain + Ollama + ChromaDB. No services, no servers (Streamlit listed in deps but not wired in current scripts).

```
pharmacy-rag-example/
├── data/                       Source documents (PDF / DOCX / TXT) — gitignored
├── chroma/                     Local vector DB (auto-built)
├── setup.py                    Environment + Ollama preflight checks
├── process_documents.py        Ingest pipeline (load → split → embed → store)
├── get_embedding_function.py   Embedding model config (default: all-MiniLM-L6-v2)
├── query_data.py               Query CLI + RAG engine
├── test_rag_system.py          Accuracy + safety benchmarks
└── requirements.txt
```

## 4. Data Model

No relational schema. Persistence is a single local Chroma collection plus the source filesystem.

| Store | Shape |
|---|---|
| **`data/`** (filesystem) | Source corpus organized in subfolders by clinical domain (e.g. cardiovascular, antidotes, pediatric dosing) |
| **`chroma/`** (ChromaDB) | One collection of chunked documents with embeddings |

**Per-chunk metadata** (set in [process_documents.py](process_documents.py)):
- `source` — full file path
- `id` — basename of the source file (used as citation key)
- `title` — display title
- `page` / `page_number` — 1-indexed page for PDFs (`PyPDFLoader`)
- `chunk_index` — sequential index across the corpus

**Retrieval result shape** (from [query_data.py](query_data.py)):
```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    { "id": "drug-monograph.pdf", "title": "...", "location": "Page 12",
      "relevance_score": 0.78, "distance": 0.28, "text_sample": "..." }
  ],
  "error": null
}
```

## 5. Key Flows

1. **Setup:** `python setup.py` → checks Python, Ollama, model availability, pulls `llama3.2:3b` if missing.
2. **Ingest:** Drop docs in `data/` → `python process_documents.py` → recursive glob (PDF / DOCX / TXT) → `RecursiveCharacterTextSplitter` (chunk 1000, overlap 100) → Chroma `from_documents()` with `all-MiniLM-L6-v2` embeddings → persisted at `chroma/`.
3. **Query:** `python query_data.py "What are the contraindications for beta-blockers?"` → similarity_search_with_score (k configurable) → context assembled with `[SOURCE_ID]` citations → Ollama LLM (`llama3.2:3b` default) prompted with a strict "answer only from context, else say you don't know" template.
4. **Benchmark:** `python test_rag_system.py` → keyword-relevance scoring, safety-critical query set, response-time metrics.

## 6. Integrations

LangChain (community / core / text-splitters / chroma / ollama / huggingface) · ChromaDB (local persistence) · Ollama (LLM runtime — default `llama3.2:3b`) · HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) · `pypdf` + `docx2txt` for ingestion. Streamlit declared in requirements but not yet integrated.

## 7. Non-Functional

- **Privacy first**: No outbound network calls; intended for HIPAA / SOC 2 / GDPR / 21 CFR Part 11 environments.
- **Reproducible ingestion**: `store_documents` deletes and rebuilds the Chroma directory on every run — deterministic but destructive; incremental indexing is a known gap.
- **Citations are mandatory**: prompt template instructs the model to refuse rather than hallucinate.
- **Safety benchmarks** must pass before any clinical-adjacent use.
- **Hardware budget**: 8 GB RAM minimum, 16 GB recommended; 10 GB free disk for models + index.
