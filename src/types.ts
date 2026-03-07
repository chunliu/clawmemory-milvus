/**
 * Type definitions for memory-milvus plugin
 */

export type MemorySource = "memory" | "sessions";

export interface Chunk {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  text: string;
  embedding: number[];
  model: string;
  source: MemorySource;
}

export interface MemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: MemorySource;
  citation?: string;
}

export interface MemorySyncProgressUpdate {
  completed: number;
  total: number;
  label?: string;
}

export interface MemoryProviderStatus {
  backend: "milvus";
  provider: string;
  model?: string;
  requestedProvider?: string;
  files?: number;
  chunks?: number;
  dirty?: boolean;
  workspaceDir?: string;
  collectionName?: string;
  address?: string;
  sources?: MemorySource[];
  sourceCounts?: Array<{ source: MemorySource; files: number; chunks: number }>;
  cache?: { enabled: boolean; entries?: number; maxEntries?: number };
  fts?: { enabled: boolean; available: boolean; error?: string };
  vector?: {
    enabled: boolean;
    available?: boolean;
    dims?: number;
  };
  hybrid?: {
    enabled: boolean;
    vectorWeight: number;
    textWeight: number;
  };
  mmr?: {
    enabled: boolean;
    lambda: number;
  };
  temporalDecay?: {
    enabled: boolean;
    halfLifeDays: number;
  };
  custom?: Record<string, unknown>;
}

export type EmbeddingProvider =
  | "openai"
  | "gemini"
  | "voyage"
  | "mistral"
  | "ollama"
  | "auto";

export interface EmbeddingProviderResult {
  provider: EmbeddingProvider;
  model: string;
  requestedProvider: EmbeddingProvider;
  providerUnavailableReason?: string;
  fallbackFrom?: EmbeddingProvider;
  fallbackReason?: string;
}
