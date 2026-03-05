import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import { watch, FSWatcher } from 'chokidar';
import { debounce } from 'lodash';
import * as fs from 'fs/promises';
import * as path from 'path';
import {
  MilvusConfig,
  MemoryChunk,
  SearchResult,
  SearchOptions,
  CollectionSchema,
  MetadataFilters,
} from './types';

/**
 * Milvus-based memory backend for OpenClaw
 */
export class MilvusMemoryBackend {
  private client: MilvusClient | null = null;
  private config: MilvusConfig;
  private watcher: FSWatcher | null = null;
  private collectionName: string;
;

  constructor(config: MilvusConfig) {
    this.config = config;
    this.collectionName = config.collection;
  }

  /**
   * Initialize Milvus connection and create collection if needed
   */
  async init(): Promise<void> {
    const address = `${this.config.host}:${this.config.port}`;
    this.client = new MilvusClient({ address });

    // Check if collection exists
    const hasCollection = await this.client.hasCollection({
      collection_name: this.collectionName,
    });

    if (!hasCollection) {
      await this.createCollection();
    }

    // Start file watcher if enabled
    if (this.config.sync.watch) {
      this.startWatcher();
    }
  }

  /**
   * Create Milvus collection with schema
   */
  private async createCollection(): Promise<void> {
    if (!this.client) throw new Error('Client not initialized');

    const schema = this.getSchema();

    await this.client.createCollection({
      collection_name: this.collectionName,
      fields: schema.fields.map(field => ({
        name: field.name,
        description: field.description || '',
        data_type: this.mapDataType(field.type),
        max_length: field.max_length,
        dim: field.dim,
        element_type: field.element_type ? this.mapDataType(field.element_type) : undefined,
        is_primary_key: field.is_primary,
        autoID: field.autoID,
      })),
      enable_dynamic_field: true,
    });

    // Create index for vector field
    await this.client.createIndex({
      collection_name: this.collectionName,
      field_name: 'embedding',
      index_type: 'IVF_FLAT',
      metric_type: this.config.search.metricType,
      params: { nlist: 128 },
    });

    // Load collection into memory
    await this.client.loadCollection({
      collection_name: this.collectionName,
    });
  }

  /**
   * Get collection schema
   */
  private getSchema(): CollectionSchema {
    return {
      name: this.collectionName,
      description: 'OpenMemory vectors',
      fields: [
        {
          name: 'id',
          type: 'VARCHAR',
          max_length: 100,
          is_primary: true,
          autoID: false,
        },
        {
          name: 'text',
          type: 'VARCHAR',
          max_length: 65535,
        },
        {
          name: 'embedding',
          type: 'FLOAT_VECTOR',
          dim: this.config.embedding.dimension,
        },
        {
          name: 'path',
          type: 'VARCHAR',
          max_length: 500,
        },
        {
          name: 'line_start',
          type: 'INT64',
        },
        {
          name: 'line_end',
          type: 'INT64',
        },
        {
          name: 'timestamp',
          type: 'INT64',
        },
        {
          name: 'tags',
          type: 'ARRAY',
          element_type: 'VARCHAR',
        },
      ],
    };
  }

  /**
   * Map data type string to Milvus DataType enum
   */
  private mapDataType(type: string): DataType {
    const typeMap: Record<string, DataType> = {
      VARCHAR: DataType.VarChar,
      INT64: DataType.Int64,
      FLOAT_VECTOR: DataType.FloatVector,
      ARRAY: DataType.Array,
    };
    return typeMap[type] || DataType.VarChar;
  }

  /**
   * Index memory files
   */
  async indexFiles(files: string[]): Promise<void> {
    if (!this.client) throw new Error('Client not initialized');

    const chunks: MemoryChunk[] = [];

    for (const file of files) {
      const fileChunks = await this.chunkFile(file);
      chunks.push(...fileChunks);
    }

    await this.insertChunks(chunks);
  }

  /**
   * Chunk a Markdown file into searchable pieces
   */
  private async chunkFile(filePath: string): Promise<MemoryChunk[]> {
    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.split('\n');
    const chunks: MemoryChunk[] = [];

    // Simple chunking strategy: ~400 tokens per chunk with overlap
    const chunkSize = 400;
    const overlap = 80;
    let currentChunk: string[] = [];
    let lineStart = 0;

    for (let i = 0; i < lines.length; i++) {
      currentChunk.push(lines[i]);

      if (currentChunk.length >= chunkSize) {
        const chunkText = currentChunk.join('\n');
        chunks.push({
          id: `${filePath}:${lineStart}-${i}`,
          text: chunkText,
          path: filePath,
          lineStart,
          lineEnd: i,
          timestamp: Date.now(),
        });

        // Keep overlap for next chunk
        currentChunk = currentChunk.slice(-overlap);
        lineStart = i - overlap + 1;
      }
    }

    // Add remaining lines
    if (currentChunk.length > 0) {
      chunks.push({
        id: `${filePath}:${lineStart}-${lines.length - 1}`,
        text: currentChunk.join('\n'),
        path: filePath,
        lineStart,
        lineEnd: lines.length - 1,
        timestamp: Date.now(),
      });
    }

    return chunks;
  }

  /**
   * Insert chunks into Milvus
   */
  private async insertChunks(chunks: MemoryChunk[]): Promise<void> {
    if (!this.client) throw new Error('Client not initialized');
    if (chunks.length === 0) return;

    // Generate embeddings for all chunks
    const embeddings = await this.generateEmbeddings(
      chunks.map(c => c.text)
    );

    // Prepare data for insertion
    const data = {
      ids: chunks.map(c => c.id),
      texts: chunks.map(c => c.text),
      embeddings,
      paths: chunks.map(c => c.path),
      line_starts: chunks.map(c => c.lineStart),
      line_ends: chunks.map(c => c.lineEnd),
      timestamps: chunks.map(c => c.timestamp),
      tags: chunks.map(c => c.tags || []),
    };

    await this.client.insert({
      collection_name: this.collectionName,
      data,
    });

    // Flush to ensure data is persisted
    await this.client.flush({
      collection_names: [this.collectionName],
    });
  }

  /**
   * Generate embeddings for text
   */
  private async generateEmbeddings(texts: string[]): Promise<number[][]> {
    // TODO: Implement embedding generation based on config
    // This is a placeholder - actual implementation will call
    // OpenAI, Gemini, or local embedding service
    
    // For now, return dummy embeddings
    return texts.map(() => 
      Array(this.config.embedding.dimension).fill(0).map(() => Math.random())
    );
  }

  /**
   * Search for similar memories
   */
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!this.client) throw new Error('Client not initialized');

    // Generate query embedding
    const queryEmbedding = await this.generateEmbeddings([query]);
    
    // Build search expression from filters
    const expr = this.buildFilterExpression(options.filters);

    // Search Milvus
    const searchResult = await this.client.search({
      collection_name: this.collectionName,
      data: [queryEmbedding[0]],
      limit: options.topK || this.config.search.topK,
      output_fields: ['text', 'path', 'line_start', 'line_end', 'timestamp', 'tags'],
      filter: expr || undefined,
    });

    // Transform results
    return searchResult.results[0].map((result: any) => ({
      id: result.id,
      text: result.entity.text,
      path: result.entity.path,
      lineStart: result.entity.line_start,
      lineEnd: result.entity.line_end,
      score: result.score,
      metadata: {
        timestamp: result.entity.timestamp,
        tags: result.entity.tags,
      },
    }));
  }

  /**
   * Build filter expression from metadata filters
   */
  private buildFilterExpression(filters?: MetadataFilters): string | null {
    if (!filters) return null;

    const conditions: string[] = [];

    if (filters.path) {
      conditions.push(`path == "${filters.path}"`);
    }

    if (filters.pathPattern) {
      conditions.push(`path like "${filters.pathPattern}"`);
    }

    if (filters.minTimestamp) {
      conditions.push(`timestamp >= ${filters.minTimestamp}`);
    }

    if (filters.maxTimestamp) {
      conditions.push(`timestamp <= ${filters.maxTimestamp}`);
    }

    if (filters.tags && filters.tags.length > 0) {
      const tagConditions = filters.tags.map(tag => `"${tag}" in tags`);
      conditions.push(`(${tagConditions.join(' or ')})`);
    }

    return conditions.length > 0 ? conditions.join(' and ') : null;
  }

  /**
   * Start file watcher
   */
  private startWatcher(): void {
    const workspace = process.env.OPENCLAW_WORKSPACE || process.cwd();
    const memoryDir = path.join(workspace, 'memory');

    this.watcher = watch([memoryDir, path.join(workspace, 'MEMORY.md')], {
      ignored: /(^|[\/\\])\../,
      persistent: true,
    });

    const debouncedIndex = debounce(async () => {
      const files = await this.getMemoryFiles();
      await this.indexFiles(files);
    }, this.config.sync.debounceMs);

    this.watcher.on('change', debouncedIndex);
    this.watcher.on('add', debouncedIndex);
  }

  /**
   * Get all memory files
   */
  private async getMemoryFiles(): Promise<string[]> {
    const workspace = process.env.OPENCLAW_WORKSPACE || process.cwd();
    const files: string[] = [];

    // Add MEMORY.md if it exists
    const memoryMd = path.join(workspace, 'MEMORY.md');
    try {
      await fs.access(memoryMd);
      files.push(memoryMd);
    } catch {
      // File doesn't exist, skip
    }

    // Add all files in memory/ directory
    const memoryDir = path.join(workspace, 'memory');
    try {
      const entries = await fs.readdir(memoryDir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isFile() && entry.name.endsWith('.md')) {
          files.push(path.join(memoryDir, entry.name));
        }
      }
    } catch {
      // Directory doesn't exist, skip
    }

    return files;
  }

  /**
   * Read a memory file
   */
  async readFile(filePath: string): Promise<string> {
    return fs.readFile(filePath, 'utf-8');
  }

  /**
   * Close connection and cleanup
   */
  async close(): Promise<void> {
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = null;
    }

    if (this.client) {
      await this.client.closeConnection();
      this.client = null;
    }
  }
}
