/** Mirrors server/setu/harness/contracts.py. Keep the two in sync. */

export type Layer = "atomic" | "semantic" | "window" | "document";
export type Decision = "allow" | "flag" | "block";
export type TurnStatus = "answered" | "abstained" | "refused" | "degraded" | "error";
export type AnswerMode = "reflex" | "deliberate" | "auto";

export interface LayerPlan {
  layer: Layer;
  k: number;
  weight: number;
}

export interface RoutePlan {
  shape: string;
  script: string;
  lang: string;
  layers: LayerPlan[];
  dense_weight: number;
  sparse_weight: number;
  rerank: boolean;
  expand_parents: boolean;
  token_count: number;
  reason: string;
}

export interface ChunkHit {
  chunk_id: string;
  doc_id: string;
  layer: Layer;
  text: string;
  lang: string;
  score: number;
  dense_score: number | null;
  sparse_score: number | null;
  rerank_score: number | null;
  rrf: number | null;
  rank_dense: number | null;
  rank_sparse: number | null;
  parent_id: string | null;
  char_start: number;
  char_end: number;
  meta: Record<string, unknown>;
}

export interface RetrievalResult {
  hits: ChunkHit[];
  dense_candidates: number;
  sparse_candidates: number;
  fused_candidates: number;
  layers_hit: Layer[];
  top_score: number;
  margin: number;
  truncated_by_budget: boolean;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  doc_id: string;
  layer: Layer;
  snippet: string;
  score: number;
}

export interface Claim {
  text: string;
  support: number;
  cited: number[];
  numeric_ok: boolean;
}

export interface GuardVerdict {
  gate: string;
  stage: "input" | "retrieval" | "output";
  decision: Decision;
  score: number;
  threshold: number | null;
  reason: string;
  latency_ms: number;
  detail: Record<string, unknown>;
}

export interface GuardReport {
  verdicts: GuardVerdict[];
  blocked: boolean;
  blocked_by: string | null;
  trust_score: number;
  groundedness: number;
  safe_message: string | null;
}

export interface SpanModel {
  name: string;
  depth: number;
  start_ms: number;
  duration_ms: number;
  status: string;
  attrs: Record<string, unknown>;
}

export interface BudgetReport {
  total_ms: number;
  used_ms: number;
  remaining_ms: number;
  slo_met: boolean;
  skipped: string[];
  degraded: boolean;
}

export interface TranscriptResult {
  text: string;
  lang: string | null;
  lang_confidence: number | null;
  provider: string;
  model: string;
  duration_ms: number;
  words: Array<Record<string, unknown>>;
}

export interface TurnResult {
  turn_id: string;
  status: TurnStatus;
  query: string;
  normalized_query: string;
  answer: string;
  answer_lang: string;
  path: string;
  model: string;
  citations: Citation[];
  claims: Claim[];
  evidence: ChunkHit[];
  route: RoutePlan | null;
  retrieval: RetrievalResult | null;
  guards: GuardReport;
  trust_score: number;
  transcript: TranscriptResult | null;
  spans: SpanModel[];
  budget: BudgetReport | null;
  pipeline_ms: number;
  stt_ms: number;
  e2e_ms: number;
  slo_met: boolean;
  warnings: string[];
  speculative_hit: boolean;
}

export interface Percentiles {
  p50: number;
  p70?: number;
  p90?: number;
  p95?: number;
  p99?: number;
  p100: number;
  mean?: number;
  min?: number;
  stdev?: number;
}

export interface MetricsReport {
  samples: number;
  total_turns: number;
  window?: number;
  slo_ms: number;
  slo_compliance?: number;
  slo_breaches?: number;
  degraded_turns?: number;
  pipeline_ms: Percentiles;
  stt_ms?: Percentiles | null;
  e2e_ms?: Percentiles | null;
  stages?: Record<string, { samples: number; mean: number; p50: number; p95: number; p100: number }>;
  status_counts?: Record<string, number>;
  path_counts?: Record<string, number>;
  shape_counts?: Record<string, number>;
  lang_counts?: Record<string, number>;
  answer_rate?: number;
  mean_trust?: number;
  pipeline_histogram?: Array<{ lo: number; hi: number | null; count: number }>;
  note?: string;
}

export interface LangInfo {
  code: string;
  name: string;
}

export interface AppConfig {
  app: string;
  tagline: string;
  version: string;
  slo_ms: number;
  stt: { provider: string; enabled: boolean; model: string };
  tts: { enabled: boolean; model: string };
  deliberate: { enabled: boolean; model: string };
  /** Languages that actually have passages in the index. */
  languages: LangInfo[];
  /** Every language SETU can be asked in, with a coverage flag. */
  supported_languages: Array<LangInfo & { in_corpus: boolean }>;
  modes: AnswerMode[];
  budget: Record<string, number>;
  thresholds: Record<string, number>;
}

export interface CorpusStats {
  n_docs: number;
  n_chunks: number;
  n_sentences: number;
  dim: number;
  embed_model: string;
  embed_backend: string;
  per_layer: Record<string, number>;
  per_lang: Record<string, number>;
  chunking: Record<string, unknown>;
  languages: LangInfo[];
  layers: Array<{ layer: Layer; chunks: number; avg_chars: number }>;
  vector_store: Record<string, unknown>;
  bm25: Record<string, unknown>;
  avg_chunk_chars: number;
}

export interface HealthReport {
  status: "ready" | "warming" | "error";
  version: string;
  error: string | null;
  warmup: Record<string, unknown>;
  detail: Record<string, unknown> | null;
}

export interface SampleQuery {
  query: string;
  lang: string;
  lang_name: string;
  query_type: string | null;
  query_id?: number;
  passages?: number;
  /** The corpus contains evidence for this question - safe to offer as an example. */
  answerable?: boolean;
  /** Stronger still: a gold answer and gold passage ids exist for it. */
  has_gold?: boolean;
}

export interface QuestionBank {
  questions: SampleQuery[];
  total: number;
  returned: number;
  answerable_by_lang: Record<string, number>;
  languages: { code: string; name: string; answerable: number }[];
}

export type LogLevel = "info" | "ok" | "warn" | "err" | "spec";

export interface LogLine {
  id: number;
  at: string;
  level: LogLevel;
  text: string;
}
