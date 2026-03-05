import { Type } from "@sinclair/typebox";

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_CAPTURE_MAX_CHARS = 500;
export const DEFAULT_MILVUS_HOST = "localhost";
export const DEFAULT_MILVUS_PORT = 19530;
export const DEFAULT_COLLECTION_NAME = "openclaw_memory";
export const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";

export const MEMORY_CATEGORIES = [
  "preference",
  "decision",
  "entity",
  "fact",
  "other",
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

// ============================================================================
// Config Schema
// ============================================================================

export const memoryConfigSchema = Type.Object({
  embedding: Type.Object({
    apiKey: Type.String(),
    model: Type.Optional(Type.String()),
    baseUrl: Type.Optional(Type.String()),
    dimensions: Type.Optional(Type.Number()),
  }),
  milvus: Type.Optional(
    Type.Object({
      host: Type.Optional(Type.String()),
      port: Type.Optional(Type.Number()),
      collection: Type.Optional(Type.String()),
    })
  ),
  autoCapture: Type.Optional(Type.Boolean()),
  autoRecall: Type.Optional(Type.Boolean()),
  captureMaxChars: Type.Optional(Type.Number()),
});

export type MemoryConfig = typeof memoryConfigSchema.static;

// ============================================================================
// Utilities
// ============================================================================

export function vectorDimsForModel(model: string): number {
  const dims: Record<string, number> = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
  };
  return dims[model] ?? 1536;
}
