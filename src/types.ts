/**
 * Configuration for Milvus memory backend
 */
export interface MilvusConfig {
  host: string;
  port: number;
  collection: string;
  embedding: EmbeddingConfig;
  sync: SyncConfig;
  search: SearchConfig;
}

export interface EmbeddingConfig {
  provider: 'openai' | 'gemini' | 'local' | 'ollama';
  model: string;
  dimension: number;
  apiKey?: string;
  baseUrl?: string;
}

export interface SyncConfig {
  watch: boolean;
  interval: string; // cron-like or duration string
  debounceMs: number;
}

export interface SearchConfig {
  topK: number;
  metricType: 'COSINE' | 'L2' | 'IP';
}

/**
 * Memory chunk for indexing
 */
export interface MemoryChunk {
  id: string;
  text: string;
  path: string;
  lineStart: number;
  lineEnd: number;
  timestamp: number;
  tags?: string[];
}

/**
 * Search result
 */
export interface SearchResult {
  id: string;
  text: string;
  path: string;
  lineStart: number;
  lineEnd: number;
  score: number;
  metadata?: Record<string, any>;
}

/**
 * Search options
 */
export interface SearchOptions {
  topK?: number;
  filters?: MetadataFilters;
  includeMetadata?: boolean;
}

export interface MetadataFilters {
  path?: string;
  pathPattern?: string;
  tags?: string[];
  minTimestamp?: number;
  maxTimestamp?: number;
}

/**
 * Milvus collection schema
 */
export interface CollectionSchema {
  name: string;
  description: string;
  fields: FieldSchema[];
}

export interface FieldSchema {
  name: string;
  type: 'VARCHAR' | 'INT64' | 'FLOAT_VECTOR' | 'ARRAY';
  description?: string;
  max_length?: number;
  dim?: number;
  element_type?: 'VARCHAR';
  is_primary?: boolean;
  autoID?: boolean;
}
