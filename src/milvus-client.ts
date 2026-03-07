/**
 * Milvus client wrapper for memory storage and search
 */

import { MilvusClient, type DataType } from "@zilliz/milvus2-sdk-node";
import crypto from "node:crypto";
import type { Chunk, MemorySource, MemoryProviderStatus } from "./types.js";

export interface MilvusClientConfig {
  address: string;
  collectionName: string;
  vectorDim: number;
  username?: string;
  password?: string;
}

export class MilvusMemoryManager {
  private client: MilvusClient;
  private collectionName: string;
  private vectorDim: number;
  private initialized = false;
  private dirty = false;
  private syncing: Promise<void> | null = null;
  private stats = {
    files: 0,
    chunks: 0,
    sourceCounts: new Map<MemorySource, { files: number; chunks: number }>(),
  };

  constructor(config: MilvusClientConfig) {
    this.client = new MilvusClient({
      address: config.address,
      username: config.username,
      password: config.password,
    });
    this.collectionName = config.collectionName;
    this.vectorDim = config.vectorDim;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Check if collection exists
      const hasCollection = await this.client.hasCollection({
        collection_name: this.collectionName,
      });

      if (!hasCollection.value) {
        await this.createCollection();
      }

      // Load collection into memory
      await this.client.loadCollection({
        collection_name: this.collectionName,
      });

      this.initialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize Milvus: ${error}`);
    }
  }

  private async createCollection(): Promise<void> {
    // Create collection schema
    await this.client.createCollection({
      collection_name: this.collectionName,
      fields: [
        {
          field_name: "id",
          description: "Chunk ID",
          data_type: DataType.VarChar,
          type_params: {
            max_length: 64,
          },
          is_primary_key: true,
          autoID: false,
        },
        {
          field_name: "path",
          description: "File path",
          data_type: DataType.VarChar,
          type_params: {
            max_length: 512,
          },
        },
        {
          field_name: "start_line",
          description: "Start line number",
          data_type: DataType.Int64,
        },
        {
          field_name: "end_line",
          description: "End line number",
          data_type: DataType.Int64,
        },
        {
          field_name: "text",
          description: "Chunk text",
          data_type: DataType.VarChar,
          type_params: {
            max_length: 65535,
            enable_analyzer: true, // Enable BM25
            enable_match: true,
          },
        },
        {
          field_name: "embedding",
          description: "Vector embedding",
          data_type: DataType.FloatVector,
          type_params: {
            dim: this.vectorDim,
          },
        },
        {
          field_name: "model",
          description: "Embedding model",
          data_type: DataType.VarChar,
          type_params: {
            max_length: 64,
          },
        },
        {
          field_name: "source",
          description: "Memory source: memory or sessions",
          data_type: DataType.VarChar,
          type_params: {
            max_length: 16,
          },
        },
      ],
      enable_dynamic_field: false,
    });

    // Create vector index
    await this.client.createIndex({
      collection_name: this.collectionName,
      field_name: "embedding",
      index_type: "HNSW",
      metric_type: "COSINE",
      params: {
        M: 16,
        efConstruction: 256,
      },
    });
  }

  async insert(chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return;

    await this.client.insert({
      collection_name: this.collectionName,
      data: chunks.map((chunk) => ({
        id: chunk.id,
        path: chunk.path,
        start_line: chunk.startLine,
        end_line: chunk.endLine,
        text: chunk.text,
        embedding: chunk.embedding,
        model: chunk.model,
        source: chunk.source,
      })),
    });

    await this.client.flush({ collection_names: [this.collectionName] });

    // Update stats
    const uniquePaths = new Set(chunks.map((c) => c.path));
    this.stats.chunks += chunks.length;
    this.stats.files += uniquePaths.size;

    for (const chunk of chunks) {
      const counts = this.stats.sourceCounts.get(chunk.source) || {
        files: 0,
        chunks: 0,
      };
      counts.chunks += 1;
      this.stats.sourceCounts.set(chunk.source, counts);
    }

    for (const path of uniquePaths) {
      const source = chunks.find((c) => c.path === path)?.source;
      if (source) {
        const counts = this.stats.sourceCounts.get(source) || {
          files: 0,
          chunks: 0,
        };
        counts.files += 1;
        this.stats.sourceCounts.set(source, counts);
      }
    }
  }

  async deleteByPath(path: string): Promise<void> {
    await this.client.delete({
      collection_name: this.collectionName,
      filter: `path == "${path}"`,
    });
  }

  async searchVector(
    queryVector: number[],
    limit: number,
    model?: string,
  ): Promise<any[]> {
    const filter = model ? `model == "${model}"` : undefined;

    const results = await this.client.search({
      collection_name: this.collectionName,
      vector: queryVector,
      limit,
      output_fields: ["id", "path", "start_line", "end_line", "text", "source", "model"],
      filter,
    });

    return results.data.map((result: any) => ({
      id: result.id,
      path: result.path,
      startLine: result.start_line,
      endLine: result.end_line,
      text: result.text,
      source: result.source,
      score: 1 - result.distance, // Convert to similarity score
    }));
  }

  async searchBM25(query: string, limit: number, model?: string): Promise<any[]> {
    const filter = model ? `model == "${model}"` : undefined;

    const results = await this.client.search({
      collection_name: this.collectionName,
      data: [query],
      anns_field: "text", // BM25 search
      limit,
      output_fields: ["id", "path", "start_line", "end_line", "text", "source", "model"],
      filter,
      search_params: {
        metric_type: "BM25",
      },
    });

    return results.data.map((result: any) => ({
      id: result.id,
      path: result.path,
      startLine: result.start_line,
      endLine: result.end_line,
      text: result.text,
      source: result.source,
      score: 1 / (1 + result.distance), // BM25 rank to score
    }));
  }

  async hybridSearch(params: {
    queryVector: number[];
    queryText: string;
    vectorWeight: number;
    textWeight: number;
    limit: number;
    model?: string;
  }): Promise<any[]> {
    const filter = params.model ? `model == "${params.model}"` : undefined;

    // Milvus Hybrid Search API
    const results = await this.client.hybrid_search({
      reqs: [
        {
          req: {
            data: [params.queryVector],
            anns_field: "embedding",
            param: {
              metric_type: "COSINE",
              params: { nprobe: 10 },
            },
            limit: params.limit * 2,
            filter,
          },
          ranker: {
            type: "nnra",
            params: { k: params.limit * 2 },
          },
        },
        {
          req: {
            data: [params.queryText],
            anns_field: "text",
            param: {
              metric_type: "BM25",
            },
            limit: params.limit * 2,
            filter,
          },
          ranker: {
            type: "nnra",
            params: { k: params.limit * 2 },
          },
        },
      ],
      rerank: {
        type: "weighted_ranker",
        params: {
          weights: [params.vectorWeight, params.textWeight],
        },
      },
      limit: params.limit,
      output_fields: ["id", "path", "start_line", "end_line", "text", "source", "model"],
    });

    return results.data.map((result: any) => ({
      id: result.id,
      path: result.path,
      startLine: result.start_line,
      endLine: result.end_line,
      text: result.text,
      source: result.source,
      score: result.score,
    }));
  }

  async count(): Promise<number> {
    const stats = await this.client.getCollectionStats({
      collection_name: this.collectionName,
    });
    return parseInt(stats.row_count || "0", 10);
  }

  markDirty(): void {
    this.dirty = true;
  }

  isDirty(): boolean {
    return this.dirty;
  }

  clearDirty(): void {
    this.dirty = false;
  }

  getStatus(): MemoryProviderStatus {
    return {
      backend: "milvus",
      provider: "milvus",
      files: this.stats.files,
      chunks: this.stats.chunks,
      dirty: this.dirty,
      collectionName: this.collectionName,
      sources: ["memory", "sessions"],
      sourceCounts: Array.from(this.stats.sourceCounts.entries()).map(([source, counts]) => ({
        source,
        files: counts.files,
        chunks: counts.chunks,
      })),
      vector: {
        enabled: true,
        available: true,
        dims: this.vectorDim,
      },
      fts: {
        enabled: true,
        available: true,
      },
    };
  }

  async close(): Promise<void> {
    // Release collection from memory
    try {
      await this.client.releaseCollection({
        collection_name: this.collectionName,
      });
    } catch {
      // Ignore errors during close
    }
  }
}
