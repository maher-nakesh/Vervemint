"""Central configuration for the Industrial Maintenance Copilot.

Every other module imports its settings from here instead of hardcoding
paths, model names, or thresholds. This is the single place you change
when swapping a model or moving to a new machine.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths -------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    dataset_dir: Path = data_dir / "industrial-instruction-dataset"
    corpus_dir: Path = dataset_dir / "panasonic_v0_0"
    qa_dir: Path = dataset_dir / "panasonic_qa_v1"
    index_dir: Path = project_root / "data" / "index"
    trace_log: Path = project_root / "data" / "traces.jsonl"

    # --- Chunking ------------------------------------------------------
    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 150

    # --- Retrieval -----------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60  # standard constant for Reciprocal Rank Fusion

    # --- Generation / guardrails ----------------------------------------
    llm_model: str = "qwen2.5-7b-gpu:latest"
    min_rerank_score: float = 0.15  # below this, the answer must abstain
    max_query_chars: int = 1000
    max_agent_steps: int = 4

    class Config:
        env_prefix = "COPILOT_"


settings = Settings()
