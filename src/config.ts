/**
 * Configuration schema for memory-milvus plugin
 */

import { Type } from "@sinclair/typebox";
import type { EmbeddingProvider } from "./types.js";

export const milvusConfigSchema = Type.Object({
  milvus: Type.Object({
    address: Type.String({
      description: "Milvus server address (e.g., 'localhost:19530')",
      default: "localhost:19530",
    }),
    collectionName: Type.String({
      description: "Milvus collection name",
      default: "openclaw_memory",
    }),
    username: Type.Optional(Type.String({
      description: "Milvus username (if auth enabled)",
    })),
    password: Type.Optional(Type.String({
      description: "Milvus password (if auth enabled)",
    })),
  }),
  embedding: Type.Object({
    provider: Type.Union(
      [
        Type.Literal("openai"),
        Type.Literal("gemini"),
        Type.Literal("voyage"),
        Type.Literal("mistral"),
        Type.Literal("ollama"),
        Type.Literal("auto"),
      ],
      {
        description: "Embedding provider",
        default: "auto",
      },
    ),
    model: Type.String({
      description: "Embedding model name",
      default: "text-embedding-3-small",
    }),
    apiKey: Type.Optional(Type.String({
      description: "API key for embedding provider",
    })),
    baseUrl: Type.Optional(Type.String({
      description: "Base URL for embedding provider",
    })),
    dimensions: Type.Optional(Type.Number({
      description: "Embedding dimensions (auto-detected if not specified)",
    })),
  }),
  search: Type.Object({
    maxResults: Type.Number({
      description: "Max search results",
      default: 5,
    }),
    minScore: Type.Number({
      description: "Minimum similarity score (0-1)",
      default: 0.5,
    }),
    hybrid: Type.Object({
      enabled: Type.Boolean({
        description: "Enable hybrid search (vector + BM25)",
        default: true,
      }),
      vectorWeight: Type.Number({
        description: "Vector search weight",
        default: 0.7,
      }),
      textWeight: Type.Number({
        description: "BM25 search weight",
        default: 0.3,
      }),
    }),
    mmr: Type.Object({
      enabled: Type.Boolean({
        description: "Enable MMR re-ranking",
        default: false,
      }),
      lambda: Type.Number({
        description: "MMR lambda (0=max diversity, 1=max relevance)",
        default: 0.7,
      }),
    }),
    temporalDecay: Type.Object({
      enabled: Type.Boolean({
        description: "Enable temporal decay",
        default: false,
      }),
      halfLifeDays: Type.Number({
        description: "Half-life in days",
        default: 30,
      }),
    }),
  }),
  sync: Type.Object({
    onSearch: Type.Boolean({
      description: "Sync on search if dirty",
      default: true,
    }),
    onSessionStart: Type.Boolean({
      description: "Sync on session start",
      default: true,
    }),
    watch: Type.Boolean({
      description: "Watch memory files for changes",
      default: true,
    }),
    debounceMs: Type.Number({
      description: "Debounce delay for file changes (ms)",
      default: 1500,
    }),
  }),
  cache: Type.Object({
    enabled: Type.Boolean({
      description: "Enable embedding cache",
      default: true,
    }),
    maxEntries: Type.Number({
      description: "Max cache entries",
      default: 50000,
    }),
  }),
  sources: Type.Optional(
    Type.Array(
      Type.Union([Type.Literal("memory"), Type.Literal("sessions")]),
      {
        description: "Memory sources to index",
        default: ["memory"],
      },
    ),
  ),
});

export type MilvusConfig = typeof milvusConfigSchema.static;
