"""Central configuration.

Every tunable in SETU lives here so that the latency budget, the chunk lattice
geometry and the guardrail thresholds can be reasoned about in one place.
Values are overridable through environment variables (prefix ``SETU_``) or a
local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class LatencyBudgetConfig(BaseSettings):
    """Per-stage millisecond allowances.

    The sum of the *hard* stages must stay under ``total_ms``. Optional stages
    (rerank, parent expansion, deliberate generation) are skipped by the
    orchestrator when the remaining budget cannot cover them - this is what
    makes the pipeline *anytime*: it always returns the best answer it has
    by the deadline instead of blowing the SLO.
    """

    total_ms: float = 200.0
    guard_input_ms: float = 8.0
    route_ms: float = 4.0
    embed_query_ms: float = 45.0
    retrieve_dense_ms: float = 30.0
    retrieve_sparse_ms: float = 18.0
    fuse_ms: float = 6.0
    expand_ms: float = 8.0
    rerank_ms: float = 20.0
    compose_ms: float = 25.0
    guard_output_ms: float = 18.0
    reserve_ms: float = 10.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SETU_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- runtime
    app_name: str = "SETU"
    app_tagline: str = "Speech to Evidence to Truth to Utterance"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: str = "*"

    # ------------------------------------------------------------------ paths
    data_dir: Path = REPO_ROOT / "data"
    index_dir: Path = REPO_ROOT / "data" / "index"
    corpus_file: Path = REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
    sample_corpus_file: Path = REPO_ROOT / "data" / "sample" / "corpus.jsonl"
    web_dist: Path = REPO_ROOT / "web" / "dist"

    # -------------------------------------------------------------- embedding
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384
    # Small on purpose. Batches are length-bucketed (see Encoder.encode), and past ~32 the
    # residual padding waste outweighs the batching gain on this corpus. Measured: naive
    # batch_size=256 ran at 42 texts/s, length-bucketed 32 is several times faster.
    embed_batch: int = 32
    embed_cache_size: int = 4096

    # Encoder intra-op threads. 2 is not a guess - it is the measured optimum for *tail*
    # latency, and the default (one thread per core) is actively harmful here.
    #
    # ONNX Runtime's intra-op pool spins briefly and then parks. Measured on a 16-core
    # laptop, encoding one short query:
    #
    #   threads   back-to-back    with a 40 ms gap between calls
    #   default        5.8 ms                 202.3 ms
    #   1             53.0 ms                  49.7 ms
    #   2             39.9 ms                  42.8 ms
    #
    # A big pool is superb while it stays hot and catastrophic once it has parked - and
    # real traffic always has gaps, because between two queries the pipeline is doing
    # retrieval, verification and I/O. Benchmarking in a tight loop hides this completely
    # and reports the 5.8 ms figure, which no user would ever experience.
    #
    # So we pick the configuration with the best *tail* rather than the best median:
    # ~43 ms, stable, independent of arrival pattern. Set 0 to restore the
    # spin-dependent default.
    embed_threads: int = 2

    # ----------------------------------------------------------- vector store
    # "auto" resolves to qdrant when reachable, otherwise the in-process flat store.
    vector_backend: str = "auto"  # auto | qdrant | flat
    qdrant_url: str = ""
    qdrant_path: str = ""
    qdrant_collection: str = "setu_lattice"

    # ------------------------------------------------------- chunk lattice
    #
    # Calibrated against the corpus rather than copied from a tutorial. MS MARCO passages
    # are short - median ~290 characters, 3 to 5 sentences - so the usual 1000-token
    # chunk settings would make the semantic and window layers byte-identical to the
    # document layer on every single passage, and dedup would silently erase two of the
    # four views. These values put roughly one boundary decision inside a typical passage
    # and several inside a long one, so the lattice is genuinely multi-scale where there
    # is structure to find and collapses honestly to 2 layers where there is not.
    # Which lattice layers to build. Exists so scripts/ablate.py can measure what each
    # layer actually contributes instead of asserting that it helps.
    enabled_layers: str = "atomic,semantic,window,document"
    atomic_min_chars: int = 45
    atomic_max_chars: int = 240
    semantic_max_chars: int = 420
    semantic_z: float = 0.45  # boundary sensitivity (lower -> more boundaries)
    window_chars: int = 300
    window_overlap_min: float = 0.14
    window_overlap_max: float = 0.50
    metadata_prefix: bool = True

    # ---------------------------------------------------------- retrieval
    dense_k: int = 40
    sparse_k: int = 40
    fuse_k: int = 24
    rerank_k: int = 12
    final_k: int = 5
    rrf_k: float = 60.0
    dense_weight: float = 0.62
    sparse_weight: float = 0.38

    # ---------------------------------------------------------- guardrails
    topicality_min_score: float = 0.34
    topicality_min_margin: float = 0.015
    groundedness_min_coverage: float = 0.42
    groundedness_min_claim: float = 0.30
    numeric_consistency: bool = True
    trust_abstain_below: float = 45.0
    safety_similarity_threshold: float = 0.68

    # -------------------------------------------------------------- answering
    answer_mode: str = "reflex"  # reflex | deliberate | auto
    answer_max_sentences: int = 3
    answer_max_chars: int = 620

    # -------------------------------------------------------- speech services
    stt_provider: str = "elevenlabs"  # elevenlabs (task allows sarvam | elevenlabs)
    # `validation_alias` matters here. Every other field picks up the SETU_ prefix, but the
    # provider keys are conventionally unprefixed - that is how ElevenLabs and Groq document
    # them, and it is what .env.example tells you to paste. Without an explicit alias,
    # pydantic-settings would only look for SETU_ELEVENLABS_API_KEY and a correctly filled
    # .env would silently do nothing. Both spellings are accepted, unprefixed first.
    elevenlabs_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ELEVENLABS_API_KEY", "SETU_ELEVENLABS_API_KEY"),
    )
    elevenlabs_stt_model: str = "scribe_v2"
    elevenlabs_tts_model: str = "eleven_flash_v2_5"  # lowest time-to-first-byte
    # Used for languages where the API will not accept a `language_code` pin (everything
    # outside en/hi/ta on this account). Slower, but noticeably better prosody on Indic
    # scripts - and an answer read with the wrong prosody is worse than one read late.
    elevenlabs_tts_fallback_model: str = "eleven_multilingual_v2"
    # "Sarah" - one of the default voices available on every account, including free tier.
    #
    # The previous default was Rachel (21m00Tcm4TlvDq8ikWAM), which is a *shared voice
    # library* voice. Free accounts get HTTP 402 "Free users cannot use library voices via
    # the API" for those, so voice output failed on exactly the accounts most likely to be
    # trying the project for the first time. Override with SETU_ELEVENLABS_VOICE_ID.
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    stt_timeout_s: float = 20.0

    # ----------------------------------------------------- deliberate path LLM
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # The deliberate path requires strict JSON output (see setu/answer/llm.py). Verified
    # against Groq's current catalogue: `llama-3.1-8b-instant` has been retired (404), and
    # the gpt-oss / qwen reasoning models fail Groq's `response_format=json_object`
    # validation because they emit reasoning preamble before the object. `groq/compound-mini`
    # honours JSON mode cleanly and is fast. Override with SETU_LLM_MODEL.
    #
    # Re-verified 2026-11-01: `groq/compound-mini` now returns 404 model_not_found for
    # standard keys. gpt-oss-20b honours JSON mode when `reasoning_effort` is sent (see
    # answer/llm.py), answering a one-passage probe in ~0.5 s.
    llm_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GROQ_API_KEY", "OPENAI_API_KEY", "SETU_LLM_API_KEY"),
    )
    llm_timeout_s: float = 12.0
    # Reasoning models spend completion tokens on reasoning before the JSON object.
    llm_max_tokens: int = 1200
    # Model for the SETU 2.0 research styles (detailed / comparative / critical /
    # executive), business analysis and query rewriting. Verified against Groq's
    # catalogue on 2026-11-01: gpt-oss honours `response_format=json_object` when
    # `reasoning_effort` is set, and the 120B variant follows the multi-claim
    # citation contract far more reliably than the 20B one.
    research_model: str = "openai/gpt-oss-120b"
    research_fast_model: str = "openai/gpt-oss-20b"
    research_timeout_s: float = 45.0
    research_reasoning_effort: str = "low"

    # -------------------------------------------------------- SETU 2.0 storage
    # SQLite by default. Any SQLAlchemy URL works (postgresql+psycopg://...).
    database_url: str = ""
    collections_dir: Path = REPO_ROOT / "data" / "collections"
    sources_dir: Path = REPO_ROOT / "data" / "sources"

    # --------------------------------------------------------- ingestion limits
    max_upload_mb: float = 25.0
    max_files_per_upload: int = 10
    max_url_mb: float = 15.0
    url_timeout_s: float = 20.0
    url_max_redirects: int = 3
    # Long documents are split into sections before the chunk lattice runs, because
    # the encoder only reads 128 tokens and a "whole report" vector is really the
    # vector of its first paragraph. See setu/knowledge/segment.py.
    section_target_chars: int = 1800
    section_max_chars: int = 2600

    # -------------------------------------------------------------- security
    # When set, every mutating or paid route requires `Authorization: Bearer <token>`
    # (or `X-SETU-Token`). Empty = open, which is only appropriate on localhost.
    api_token: str = ""
    rate_limit_per_min: int = 60
    max_tts_chars: int = 2500
    max_audio_mb: float = 15.0

    # -------------------------------------------------------------- telemetry
    metrics_window: int = 4096
    trace_retain: int = 256
    trace_db_retain: int = 2000

    budget: LatencyBudgetConfig = Field(default_factory=LatencyBudgetConfig)

    # ------------------------------------------------------------------ helpers
    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def layer_set(self) -> set[str]:
        return {s.strip() for s in self.enabled_layers.split(",") if s.strip()}

    @property
    def stt_enabled(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'setu.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.index_dir, self.corpus_file.parent, self.collections_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
