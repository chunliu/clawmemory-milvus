# OpenClaw Memory (Milvus) Plugin

Milvus-backed memory plugin for OpenClaw with hybrid search (vector + BM25).

## Features

- **Hybrid Search**: Combines vector similarity search with BM25 keyword search
- **Milvus Backend**: Uses Milvus vector database for scalable storage and search
- **Multi-Agent Isolation**: Each agent gets its own Milvus collection
- **Multiple Embedding Providers**: Supports OpenAI, Gemini, Voyage, Mistral, Ollama
- **MMR Re-ranking**: Maximal Marginal Relevance for diverse results
- **Temporal Decay**: Boosts recent memories, decays old ones
- **File Watching**: Automatic sync when memory files change
- **Embedding Cache**: Reduces API calls with intelligent caching

## Installation

```bash
npm install openclaw-plugin-memory-milvus
```

## Configuration

Add to your OpenClaw config:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-milvus"
    },
    "entries": {
      "memory-milvus": {
        "milvus": {
          "address": "localhost:19530",
          "prefix": "openclaw-memory",
          "username": "",
          "password": ""
        },
        "embedding": {
          "provider": "openai",
          "model": "text-embedding-3-small",
          "apiKey": "your-api-key",
          "dimensions": 1536
        },
        "search": {
          "maxResults": 5,
          "minScore": 0.5,
          "hybrid": {
            "enabled": true,
            "vectorWeight": 0.7,
            "textWeight": 0.3
          },
          "mmr": {
            "enabled": false,
            "lambda": 0.7
          },
          "temporalDecay": {
            "enabled": false,
            "halfLifeDays": 30
          }
        },
        "sync": {
          "onSearch": true,
          "onSessionStart": true,
          "watch": true,
          "debounceMs": 1500
        },
        "cache": {
          "enabled": true,
          "maxEntries": 50000
        },
        "sources": ["memory"]
      }
    }
  }
}
```

## Tools

### memory_search

Search memory files using hybrid vector + BM25 search.

```typescript
{
  query: string;        // Search query
  maxResults?: number;  // Max results (default: 5)
  minScore?: number;    // Minimum similarity score (0-1)
}
```

### memory_get

Read memory file content.

```typescript
{
  path: string;    // File path relative to workspace
  from?: number;  // Start line (1-indexed)
  lines?: number; // Number of lines to read
}
```

## CLI Commands

```bash
# Show memory status
openclaw memory status

# Search memories
openclaw memory search "your query"

# Sync memory files to Milvus
openclaw memory sync
```

## Architecture

```
┌─────────────────────────────────────┐
│         OpenClaw Memory API         │
│  (memory_search, memory_get)        │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│      Milvus Memory Manager         │
│  - Agent-specific collections       │
│  - Hybrid Search                   │
│  - File Sync                       │
│  - File Watching                   │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│         Milvus Server               │
│  - Collection per agent             │
│  - Vector Search (HNSW)            │
│  - BM25 Search                      │
│  - Hybrid Search API                │
└─────────────────────────────────────┘
```

## Multi-Agent Isolation

Each OpenClaw agent gets its own Milvus collection:

- **Collection Naming**: `{prefix}-{agentId}`
- **Example**: `openclaw-memory-agent1`, `openclaw-memory-agent2`
- **Isolation**: Complete data separation between agents
- **Security**: Agents cannot access each other's memory
- **Customizable Prefix**: Configure via `milvus.prefix` in config

This mirrors OpenClaw's SQLite implementation where each agent has its own database file.

## License

MIT
