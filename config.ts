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

export type EmbeddingProvider = "openai" | "ollama" | "custom";

export type MemoryConfig = {
  embedding: {
    provider: EmbeddingProvider;
    model: string;
    apiKey?: string;
    baseUrl?: string;
    dimensions?: number;
  };
  milvus?: {
    host?: string;
    port?: number;
    collection?: string;
    username?: string;
    password?: string;
  };
  autoCapture?: boolean;
  autoRecall?: boolean;
  captureMaxChars?: number;
};

// ============================================================================
// Utilities
// ============================================================================

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  // OpenAI models
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
  // Ollama models (common defaults)
  "nomic-embed-text": 768,
  "mxbai-embed-large": 1024,
  "all-minilm": 384,
  "llama3": 4096,
  "mistral": 1024,
  "qwen3-embed": 1024,
};

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    // For unknown models, return a common default
    // User should specify dimensions explicitly for custom models
    return 1536;
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

function resolveEmbeddingProvider(provider: unknown): EmbeddingProvider {
  if (typeof provider === "string") {
    if (provider === "openai" || provider === "ollama" || provider === "custom") {
      return provider as EmbeddingProvider;
    }
  }
  return "openai"; // Default
}

function resolveEmbeddingModel(embedding: Record<string, unknown>): string {
  const model = typeof embedding.model === "string" ? embedding.model : DEFAULT_EMBEDDING_MODEL;
  return model;
}

function resolveEmbeddingBaseUrl(embedding: Record<string, unknown>, provider: EmbeddingProvider): string {
  // If baseUrl is explicitly provided, use it
  if (typeof embedding.baseUrl === "string") {
    return resolveEnvVars(embedding.baseUrl);
  }

  // Default URLs based on provider
  switch (provider) {
    case "ollama":
      return "http://localhost:11434/v1";
    case "custom":
      return "http://localhost:11434/v1";
    case "openai":
    default:
      return "https://api.openai.com/v1";
  }
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
    if (!embedding) {
      throw new Error("embedding config is required");
    }
    assertAllowedKeys(embedding, ["provider", "model", "apiKey", "baseUrl", "dimensions"], "embedding config");

    const provider = resolveEmbeddingProvider(embedding.provider);
    const model = resolveEmbeddingModel(embedding);

    // For OpenAI, apiKey is required
    if (provider === "openai" && typeof embedding.apiKey !== "string") {
      throw new Error("embedding.apiKey is required for OpenAI provider");
    }

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
        provider,
        model,
        apiKey: typeof embedding.apiKey === "string" ? resolveEnvVars(embedding.apiKey) : undefined,
        baseUrl: resolveEmbeddingBaseUrl(embedding, provider),
        dimensions: typeof embedding.dimensions === "number" ? embedding.dimensions : undefined,
      },
      milvus: cfg.milvus as MemoryConfig["milvus"],
      autoCapture: cfg.autoCapture === true,
      autoRecall: cfg.autoRecall !== false,
      captureMaxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
    };
  },
  uiHints: {
    "embedding.provider": {
      label: "Embedding Provider",
      help: "Provider for embeddings (openai, ollama, custom)",
      placeholder: "openai",
    },
    "embedding.apiKey": {
      label: "API Key",
      sensitive: true,
      placeholder: "sk-proj-...",
      help: "API key for OpenAI embeddings (required for openai provider)",
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_EMBEDDING_MODEL,
      help: "Model name (e.g., text-embedding-3-small, nomic-embed-text)",
    },
    "embedding.baseUrl": {
      label: "Base URL",
      placeholder: "https://api.openai.com/v1",
      help: "Base URL for embeddings API (default varies by provider)",
      advanced: true,
    },
    "embedding.dimensions": {
      label: "Dimensions",
      placeholder: "1536",
      help: "Vector dimensions (required for custom models)",
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
