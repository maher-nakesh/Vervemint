# Industrial Maintenance Copilot

A small-scale implementation of the Industrial Copilot pattern: a technician asks about a machine, and the system combines simulated machine-telemetry data with retrieval over industrial manuals (Panasonic datasheets and service guides). It answers with page-level citations, refuses when it has no evidence, and proposes a work order that a human must approve. Every retrieval decision is backed by an evaluation benchmark.

> **Note:** Machine telemetry is simulated. Documents are real Panasonic industrial manuals, not Schaeffler bearing documentation.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │   Streamlit UI                   │
                    │   Chat · Citations · Under-the-hood │
                    └───────────┬──────────────────────┘
                                │
                    FastAPI  ┌───┴───┐
                    /ask  /health
                                │
                    ┌───────────┴──────────────────────┐
                    │ Input Guardrails                 │
                    │ (prompt injection, off-topic)     │
                    └───────────┬──────────────────────┘
                                │
                    Agent Loop (Qwen 2.5 7B, ≤4 steps)
                    ├── tool: search_manuals
                    │     Hybrid retrieval:
                    │       BM25 (exact part numbers) ──┐
                    │       Dense (bge-small, FAISS)  ──┤─── RRF fusion ── Cross-encoder rerank ── top-5
                    ├── tool: get_machine_health
                    │     Simulated vibration/temperature telemetry
                    └── tool: create_work_order
                          Human approval required (human-in-the-loop)
                                │
                    └───────────┬──────────────────────┘
                    │ Output Guardrails                  │
                    │ (citations must exist, abstain      │
                    │  when evidence is weak)             │
                    └───────────┬──────────────────────┘
                                │
                    Trace Log (JSONL)
                    per-stage latency, retrieval scores, tool calls
```

**Design choices:**
- No LangChain / LlamaIndex — every line is explainable.
- No vector DB server — ~12k chunks fit an exact FAISS index on disk.
- No Kubernetes — single-user prototype with production practices.

## Project Structure

```
copiliot_idustrial_Rag/
├── data/
│   ├── machine_manual.txt          # Small sample manual
│   └── industrial-instruction-dataset/  [gitignored]
├── src/
│   ├── copilot/
│   │   ├── config.py                 # Settings: models, paths, top_k
│   │   ├── ingest.py                 # Parquet → clean → chunks + metadata
│   │   ├── index.py                  # Build & persist BM25 + FAISS
│   │   ├── retrieve.py               # Hybrid search, RRF, cross-encoder rerank
│   │   ├── generate.py               # Grounded prompt, citations, abstention
│   │   ├── guardrails.py             # Input/output validation
│   │   ├── agent.py                  # Tool-calling loop + 3 tools
│   │   ├── tracing.py                # JSONL per-request trace
│   │   └── api.py                    # FastAPI server
│   └── baseline_rag.py              # Original v0 baseline (for A/B comparison)
├── evaluation/
│   ├── build_eval_set.py            # 1000 QA pairs → gold page references
│   ├── eval_retrieval.py            # Recall@5, MRR by method
│   └── eval_answers.py
├── ui/
│   └── app.py                       # Streamlit chat interface
├── tests/
│   ├── test_smoke.py                # Basic import/structure tests
│   └── __init__.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Clone
git clone https://github.com/maher-nakesh/copiliot_idustrial_Rag.git
cd copiliot_idustrial_Rag

# Create virtual environment (reuses system CUDA PyTorch)
python -m venv .venv --system-site-packages
source .venv/bin/activate   # or: .venv\Scripts\activate

# Install
pip install -r requirements.txt

# Download the dataset (Panasonic industrial manuals)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Parssky/industrial-instruction-dataset',
    repo_type='dataset', local_dir='data/industrial-instruction-dataset')
"

# Start Ollama (requires a local model)
ollama pull qwen2.5-7b-gpu:latest
```

## Usage

### API

```bash
uvicorn src.copilot.api:app --reload
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is the motor overheating?", "machine_id": "P-07"}'
```

### Streamlit UI

```bash
streamlit run ui/app.py
```

### CLI (baseline)

```bash
python src/baseline_rag.py
```

### Evaluation

```bash
# Build evaluation set
python evaluation/build_eval_set.py

# Run retrieval evaluation (baseline vs BM25 vs dense vs hybrid vs hybrid+rerank)
python evaluation/eval_retrieval.py

# Evaluate answer quality
python evaluation/eval_answers.py
```

## Evaluation Framework

The evaluation set maps 1000 QA test questions to gold pages from the Panasonic manuals. Each retrieval method is scored on:

| Method | Recall@5 | MRR | Notes |
|---|---|---|---|
| v0 (baseline RAG) | _TBD_ | _TBD_ | Dense-only, no rerank |
| BM25 | _TBD_ | _TBD_ | Exact part-number match |
| Dense (bge-small) | _TBD_ | _TBD_ | Misses part numbers like ERJ-D1 |
| Hybrid (BM25 + dense RRF) | _TBD_ | _TBD_ | |
| Hybrid + rerank | _TBD_ | _TBD_ | Cross-encoder rerank |

Results are computed in Phase 3 (Evaluation). Do not quote specific numbers until measured.

## Honest Limits

- **Telemetry is simulated.** Values are generated, not live.
- **Documents are real Panasonic industrial datasheets and manuals.**
- **Runs on a local 7B model** — answer quality is below frontier models, but the architecture is model-agnostic.
- **Single-user prototype** with production practices (tracing, evals, guardrails, tests). Not proven at scale.

## Key Models

| Component | Model | Purpose |
|---|---|---|
| Dense retriever | `BAAI/bge-small-en-v1.5` | Embedding-based chunk retrieval |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder reranking of top candidates |
| LLM | `qwen2.5-7b-gpu:latest` | Grounded answer generation |
| BM25 | `bm25s` | Exact token matching (part numbers) |

## License

Educational / research use.
