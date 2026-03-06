# @openclaw/memory-milvus

Milvus-based memory backend for OpenClaw, providing long-term semantic memory with vector search.

## Features

- **Vector-based semantic search**: Find related memories even when wording differs
- **Large-scale storage**: Handle millions of memory vectors with Milvus
- **Auto-recall**: Automatically inject relevant memories into context
- **Auto-capture**: Automatically capture important information from conversations
- **Multiple embedding providers**: Support for OpenAI, Ollama, and custom providers
- **Drop-in replacement**: Compatible with OpenClaw's memory system

## Installation

```bash
npm install @openclaw/memory-milvus
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
  etetcd:
    image: quay.io/coreos/etcd:latest
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETQUOTA_BACKEND_BYTES=4294967296
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
      - milvus:/var/lib/mariadb
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

## Configuration

### Using OpenAI Embeddings

```json5
{
  "plugins": {
    "memory-milvus": {
      "embedding": {
        "provider": "openai",
        "apiKey": "${OPENAI_API_KEY}",
        "model": "text-embedding-3-small"
      },
      "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection": "openclaw_memory"
      },
      "autoCapture": true,
      "autoRecall": true,
      "captureMaxChars": 500
    }
  }
}
```

### Using Ollama (Local)

```json5
{
  "plugins": {
    "memory-milvus": {
      "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "baseUrl": "http://localhost:11434/v1"
      },
      "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection": "openclaw_memory"
      },
      "autoCapture": true,
      "autoRecall": true
    }
  }
}
```

#### Using Qwen3-Embedding-0.6B

First, download the GGUF model from HuggingFace and create an Ollama model:

```bash
# Download the GGUF file
curl -L -o Qwen3-Embedding-0.6B-f16.gguf \
  "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf"

# Create a Modelfile
cat > Modelfile << 'EOF'
FROM ./Qwen3-Embedding-0.6B-f16.gguf
PARAMETER num_ctx 2048
PARAMETER num_batch 512
TEMPLATE """
{{ .Prompt }}
"""
EOF

# Create the Ollama model
ollama create qwen3-embed -f Modelfile
```

Then configure the plugin:

```json5
{
  "plugins": {
    "memory-milvus": {
      "embedding": {
        "provider": "ollama",
        "model": "qwen3-embed",
        "baseUrl": "http://localhost:11434/v1"
      },
      "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection": "openclaw_memory"
      },
      "autoCapture": true,
      "autoRecall": true
    }
  }
}
```

### Using Custom Provider

```json5
{
  "plugins": {
    "memory-milvus": {
      "embedding": {
        "provider": "custom",
        "model": "your-model-name",
        "baseUrl": "http://your-host:port/v1",
        "dimensions": 768
      },
      "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection": "openclaw_memory"
      },
      "autoCapture": true,
      "autoRecall": true
    }
  }
}
```

## Embedding Providers

### OpenAI

- **Provider**: `openai`
- **Required**: `apiKey`
- **Default baseUrl**: `https://api.openai.com/v1`
- **Supported models**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

### Ollama

- **Provider**: `ollama`
- **Required**: None (runs locally)
- **Default baseUrl**: `http://localhost:11434/v1`
- **Common models**: `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`, `llama3`, `mistral`

**Setup Oll**ama**:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull an embedding model
ollama pull nomic-embed-text

# Verify it works
curl http://localhost:11434/api/generate -d '{
  "model": "nomic-embed-text",
  "prompt": "Hello, world!"
}'
```

### Custom

- **Provider**: `custom`
- **Required**: `baseUrl`, `dimensions`
- **Use case**: Any OpenAI-compatible embedding API

## Tools

### memory_recall

Search through long-term memories.

```json
{
  "query": "What are my preferences?",
  "limit": 5
}
```

### memory_store

Save important information in long-term memory.

```json
{
  "text": "I prefer dark mode in all apps",
  "importance": 0.8,
  "category": "preference"
}
```

### memory_forget

Delete specific memories.

```json
{
  "query": "dark mode"
}
```

Or by ID:

```json
{
  "memoryId": "550e8400-e29b-41d4-a716-446655440000"
}
```

## CLI Commands

```bash
# List all memories
openclaw milvus-mem list

# Search memories
openclaw milvus-mem search "my preferences" --limit 5

# Show statistics
openclaw milvus-mem stats
```

## Memory Categories

- `preference`: User preferences and likes/dislikes
- `decision`: Past decisions and choices
- `entity`: Names, emails, phone numbers
- `fact`: General facts and information
- `other`: Everything else

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode
npm run dev

# Run tests
npm test
```

## License

MIT
