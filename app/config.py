"""Central configuration. Everything is env-driven — nothing is hardcoded."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    # "ollama" (local), "groq" (hosted, free tier, no card) or "mistral"
    # (hosted, French). See app/llm.py.
    llm_provider: str = "ollama"

    mistral_api_key: str = ""
    llm_model: str = "mistral-small-latest"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_timeout: int = 300        # a 7B model on CPU is not fast
    # Ollama's default context is 2048 tokens. Our prompt is ~1600 (five legal
    # extracts plus the instructions), so the default SILENTLY TRUNCATES the
    # extracts — the model then cites only the first one or two because it never
    # saw the rest. This must stay above build_context's budget.
    ollama_num_ctx: int = 4096
    # Answers are 3-6 sentences. Capping output is the cheapest latency win
    # there is: generation time is roughly linear in tokens produced.
    ollama_num_predict: int = 350
    # Keep the model resident between questions, or every call pays the load
    # cost again.
    ollama_keep_alive: str = "30m"

    # Groq speaks the OpenAI chat-completions protocol, so the same code path
    # serves any OpenAI-compatible endpoint — Together, OpenRouter, a local
    # vLLM — by changing the base URL alone.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Groq retires models on a few months' notice — llama-3.3-70b-versatile,
    # the first choice here, was withdrawn in August 2026. Check the live
    # catalogue with:
    #   curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY"
    groq_model: str = "openai/gpt-oss-120b"

    # --- Embeddings ---
    # NOTE: must be multilingual. The corpus is French, the demo answers FR+EN.
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"

    # --- Reranker (cross-encoder) ---
    # Reads query and chunk together, so its score is a real relevance signal
    # rather than a cosine. This is what the guardrail thresholds.
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidates: int = 20

    # --- Vector store ---
    chroma_path: Path = BASE_DIR / "chroma_db"
    collection_name: str = "sme_hr_corpus"

    # --- Chunking (characters, not tokens — see ingestion.py) ---
    chunk_size: int = 1400
    chunk_overlap: int = 200

    # --- Retrieval / guardrails ---
    top_k: int = 5
    min_similarity: float = 0.72      # dense-only strategy (bi-encoder cosine)
    min_rerank_score: float = 0.0     # hybrid_rerank strategy (cross-encoder logit)
    max_question_length: int = 500
    rate_limit: str = "10/minute"

    # --- Paths ---
    documents_path: Path = BASE_DIR / "data" / "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
