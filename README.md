# clawmemory-milvus

Milvus-based memory backend for OpenClaw.

## Overview

This plugin provides a Milvus vector database backend for OpenClaw's memory system, enabling semantic search over Markdown-based memory files with large-scale vector retrieval capabilities.

## Features

- **Vector-based semantic search**: Find related memories even when wording differs
- **Large-scale storage**: Handle millions of memory vectors with Milvus
- **Hybrid queries**: Combine vector similarity with metadata filters
- **Markdown-first**: Original Markdown files remain the source of truth
- **Drop-in replacement**: Compatible with OpenClaw's existing memory tools (`memory_search`, `memory_get`)

## Architecture

```
Markdown Files (MEMORY.md, memory/*.md)
    ↓
Chunking + Embedding
    ↓
Milvus Vector Database
    ↓
Semantic Search (memory_search)
```

## Installation

```bash
# Clone this repository
git clone https://github.com/chunliu/clawmemory-milvus.git
cd clawmemory-milvus

# Install dependencies
npm install

# Build
npm run build
```

## Configuration

Add to your OpenClaw config:

```json5
{
  memory: {
    backend: "milvus",
    milvus: {
      host: "localhost",
      port: 19530,
      collection: "openclaw_memory",
      embedding: {
        provider: "openai",
        model: "text-embedding-3-small",
        dimension: 1536
      },
      sync: {
        watch: true,
        interval: "5m",
        debounceMs: 1500
      },
      search: {
        topK: 10,
        metricType: "COSINE"
      }
    }
  }
}
```

## Milvus Setup

### Docker (Recommended)

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

### Docker Compose

```yaml
version: '3.5'

services:
  etcd:
    image: quay.io/coreos/etcd:latest
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

volumes:
  etcd:
  minio:
  milvus:
```

## Development

```bash
# Run tests
npm test

# Watch mode
npm run dev

# Lint
npm run lint
```

## License

MIT
