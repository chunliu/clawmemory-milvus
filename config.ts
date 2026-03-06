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
// Types
// ============================================================================

export type MemoryConfig = {
  embedding: {
    provider: "openai";
    model: string;
    apiKey: string;
    baseUrl?: string;
    dimensions?: number;
  };
  milvus?: {
    host?: string;
    port?: number;
    collection?: string;
  };
  autoCapture?: boolean;
  autoRecall?: boolean;
  captureMaxChars?: number;
};

// ============================================================================
// Utilities
// ============================================================================

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
};

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(`Unsupported embedding model: ${model}`);
  }
  return dims;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function resolveEmbeddingModel(embedding: Record<string, unknown>): string {
  const model = typeof embedding.model === "string" ? embedding.model : DEFAULT_EMBEDDING_MODEL;
  if (typeof embedding.dimensions !== "number") {
    vectorDimsForModel(model);
  }
  return model;
}

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

// ============================================================================
// Config Schema
// ============================================================================

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["embedding", "milvus", "autoCapture", "autoRecall", "captureMaxChars"],
      "memory config",
    );

    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding || typeof embedding.apiKey !== "string") {
      throw new Error("embedding.apiKey is required");
    }
    assertAllowedKeys(embedding, ["apiKey", "model", "baseUrl", "dimensions"], "embedding config");

    const model = resolveEmbeddingModel(embedding);

    // Validate milvus config if present
    if (cfg.milvus) {
      const milvus = cfg.milvus as Record<string, unknown>;
      assertAllowedKeys(milvus, ["host", "port", "collection"], "milvus config");
    }

    const captureMaxChars =
      typeof cfg.captureMaxChars === "number" ? Math.floor(cfg.captureMaxChars) : undefined;
    if (
      typeof captureMaxChars === "number" &&
      (captureMaxChars < 100 || captureMaxChars > 10_000)
    ) {
      throw new Error("captureMaxChars must be between 100 and 10000");
    }

    return {
      embedding: {
        provider: "openai",
        model,
        apiKey: resolveEnvVars(embedding.apiKey),
        baseUrl:
          typeof embedding.baseUrl === "string" ? resolveEnvVars(embedding.baseUrl) : undefined,
        dimensions: typeof embedding.dimensions === "number" ? embedding.dimensions : undefined,
      },
      milvus: cfg.milvus as MemoryConfig["milvus"],
      autoCapture: cfg.autoCapture === true,
      autoRecall: cfg.autoRecall !== false,
      captureMaxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
    };
  },
  uiHints: {
    "embedding.apiKey": {
      label: "OpenAI API Key",
      sensitive: true,
      placeholder: "sk-proj-...",
      help: "API key for OpenAI embeddings (or use ${OPENAI_API_KEY})",
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_EMBEDDING_MODEL,
      help: "OpenAI embedding model to use",
    },
    "embedding.baseUrl": {
      label: "Base URL",
      placeholder: "https://api.openai.com/v1",
      help: "Base URL for compatible providers (e.g. http://localhost:11434/v1)",
      advanced: true,
    },
    "embedding.dimensions": {
      label: "Dimensions",
      placeholder: "1536",
      help: "Vector dimensions for custom models (required for non-standard models)",
      advanced: true,
    },
    "milvus.host": {
      label: "Milvus Host",
      placeholder: DEFAULT_MILVUS_HOST,
      help: "Milvus server host",
    },
    "milvus.port": {
      label: "Milvus Port",
      placeholder: String(DEFAULT_MILVUS_PORT),
      help: "Milvus server port",
    },
    "milvus.collection": {
      label: "Collection Name",
      placeholder: DEFAULT_COLLECTION_NAME,
      help: "Milvus collection name",
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information from conversations",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
    captureMaxChars: {
      label: "Capture Max Chars",
      help: "Maximum message length eligible for auto-capture",
      advanced: true,
      placeholder: String(DEFAULT_CAPTURE_MAX_CHARS),
    },
  },
};
