/**
 * Milvus Provider for Memory Storage
 *
 * Handles Milvus connection, collection management, and CRUD operations.
 * Supports both conversation memories and file memories.
 */

import { randomUUID } from "node:crypto";
import { DataType } from "@zilliz/milvus2-sdk-node";

let milvusImportPromise: Promise<typeof import("@zilliz/milvus2-sdk-node")> | null = null;

const loadMilvus = async (): Promise<typeof import("@zilliz/milvus2-sdk-node")> => {
  if (!milvusImportPromise) {
    milvusImportPromise = import("@zilliz/milvus2-sdk-node");
  }
  return await milvusImportPromise;
};

// ============================================================================
// Types
// ============================================================================

export interface ConversationMemoryEntry {
  id: string;
  text: string;
  vector: number[];
  importance: number;
  category: "preference" | "decision" | "entity" | "fact" | "other";
  createdAt: number;
}

export interface FileMemoryEntry {
  id: string;
  text: string;
  vector: number[];
  path: string;
  lineStart: number;
  lineEnd: number;
  category: "file";
  createdAt: number;
  isEvergreen: boolean;
}

export interface MemorySearchResult<T> {
  entry: T;
  score: number;
}

// ============================================================================
// MemoryDB Class
// ============================================================================

export class MemoryDB {
  private client: any = null; // MilvusClient
  private: conversationCollectionName: string;
  private fileCollectionName: string;
  private vectorDim: number;
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly host: string,
    private readonly port: number,
    collectionName: string,
    vectorDim: number,
    private readonly username?: string,
    private readonly password?: string,
  ) {
    this.conversationCollectionName = `${collectionName}_conversations`;
    this.fileCollectionName = `${collectionName}_files`;
    this.vectorDim = vectorDim;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.client) {
      return;
    }
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    const { MilvusClient } = await loadMilvus();
    const address = `${this.host}:${this.port}`;

    const clientConfig: any = { address };
    if (this.username && this.password) {
      clientConfig.username = this.username;
      clientConfig.password = this.password;
    }

    this.client = new MilvusClient(clientConfig);

    // Initialize conversation collection
    await this.ensureCollection(this.conversationCollectionName, "conversation");

    // Initialize file collection
    await this.ensureCollection(this.fileCollectionName, "file");
  }

  private async ensureCollection(collectionName: string, type: "conversation" | "file"): Promise<void> {
    const hasCollection = await this.client.hasCollection({
      collection_name: collectionName,
    });

    if (!hasCollection) {
      await this.createCollection(collectionName, type);
    }

    // Load collection into memory
    try {
      await this.client.loadCollection({
        collection_name: collectionName,
      });
    } catch (error: any) {
      if (error.message && error.message.includes("CollectionNotExists")) {
        await this.createCollection(collectionName, type);
        await this.client.loadCollection({
          collection_name: collectionName,
        });
      } else {
        throw error;
      }
    }
  }

  private async createCollection(collectionName: string, type: "conversation" | "file"): Promise<void> {
    let schema: any[];

    if (type === "conversation") {
      schema = [
        {
          name: "id",
          description: "Memory ID",
          data_type: DataType.VarChar,
          max_length: 36,
          is_primary_key: true,
          autoID: false,
        },
        {
          name: "text",
          description: "Memory text",
          data_type: DataType.VarChar,
          max_length: 65535,
        },
        {
          name: "vector",
          description: "Embedding vector",
          data_type: DataType.FloatVector,
          dim: this.vectorDim,
        },
        {
          name: "importance",
          description: "Importance score",
          data_type: DataType.Float,
        },
        {
          name: "category",
          description: "Memory category",
          data_type: DataType.VarChar,
          max_length: 20,
        },
        {
          name: "createdAt",
          description: "Creation timestamp",
          data_type: DataType.Int64,
        },
      ];
    } else {
      // File collection
      schema = [
        {
          name: "id",
          description: "Memory ID",
          data_type: DataType.VarChar,
          max_length: 36,
          is_primary_key: true,
          autoID: false,
        },
        {
          name: "text",
          description: "Memory text",
          data_type: DataType.VarChar,
          max_length: 65535,
        },
        {
          name: "vector",
          description: "Embedding vector",
          data_type: DataType.FloatVector,
          dim: this.vectorDim,
        },
        {
          name: "path",
          description: "File path",
          data_type: DataType.VarChar,
          max_length: 512,
        },
        {
          name: "lineStart",
          description: "Starting line number",
          data_type: DataType.Int64,
        },
        {
          name: "lineEnd",
          description: "Ending line number",
          data_type: DataType.Int64,
        },
        {
          name: "category",
          description: "Memory category",
          data_type: DataType.VarChar,
          max_length: 20,
        },
        {
          name: "createdAt",
          description: "Creation timestamp",
          data_type: DataType.Int64,
        },
        {
          name: "isEvergreen",
          description: "Is evergreen content",
          data_type: DataType.Bool,
        },
      ];
    }

    await this.client.createCollection({
      collection_name: collectionName,
      fields: schema,
      enable_dynamic_field: true,
    });

    // Create index for vector field
    await this.client.createIndex({
      collection_name: collectionName,
      field_name: "vector",
      index_type: "IVF_FLAT",
      metric_type: "COSINE",
      params: { nlist: 128 },
    });

    // Create text index for BM25 search (file collection only)
    if (type === "file") {
      try {
        await this.client.createIndex({
          collection_name: collectionName,
          field_name: "text",
          index_type: "BM25",
        });
      } catch (error: any) {
        // BM25 might not be available in all Milvus versions
        console.warn(`Failed to create BM25 index: ${error.message}`);
      }
    }
  }

  // ========================================================================
  // Conversation Memory Operations
  // ========================================================================

  async storeConversation(entry: Omit<ConversationMemoryEntry, "id" | "createdAt">): Promise<ConversationMemoryEntry> {
    await this.ensureInitialized();

    const fullEntry: ConversationMemoryEntry = {
      ...entry,
      id: randomUUID(),
      createdAt: Date.now(),
    };

    await this.client.insert({
      collection_name: this.conversationCollectionName,
      data: [
        {
          id: fullEntry.id,
          text: fullEntry.text,
          vector: fullEntry.vector,
          importance: fullEntry.importance,
          category: fullEntry.category,
          createdAt: fullEntry.createdAt,
        },
      ],
    });

    await this.client.flush({
      collection_names: [this.conversationCollectionName],
    });

    return fullEntry;
  }

  async searchConversations(vector: number[], limit = 5, minScore = 0.5): Promise<MemorySearchResult<ConversationMemoryEntry>[]> {
    await this.ensureInitialized();

    const results = await this.client.search({
      collection_name: this.conversationCollectionName,
      data: [vector],
      limit,
      output_fields: ["text", "importance", "category", "createdAt"],
    });

    if (!results || !results.results || results.results.length === 0) {
      return [];
    }

    const mapped = results.results.map((result: any) => ({
      entry: {
        id: result.id,
        text: result.text || "",
        vector: [],
        importance: result.importance || 0,
        category: result.category || "other",
        createdAt: result.createdAt || Date.now(),
      },
      score: result.score || 0,
    }));

    return mapped.filter((r) => r.score >= minScore);
  }

  async deleteConversation(id: string): Promise<boolean> {
    await this.ensureInitialized();
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(id)) {
      throw new Error(`Invalid memory ID format: ${id}`);
    }
    await this.client.delete({
      collection_name: this.conversationCollectionName,
      expr: `id == "${id}"`,
    });
    return true;
  }

  async countConversations(): Promise<number> {
    await this.ensureInitialized();
    const stats = await this.client.getCollectionStatistics({
      collection_name: this.conversationCollectionName,
    });
    const rowCount = stats.data?.row_count;
    if (typeof rowCount === "number") {
      return rowCount;
    }
    if (typeof rowCount === "string") {
      return parseInt(rowCount, 10);
    }
    return 0;
  }

  // ========================================================================
  // File Memory Operations
  // ========================================================================

  async storeFile(entry: Omit<FileMemoryEntry, "id" | "createdAt">): Promise<FileMemoryEntry> {
    await this.ensureInitialized();

    const fullEntry: FileMemoryEntry = {
      ...entry,
      id: randomUUID(),
      createdAt: Date.now(),
    };

    await this.client.insert({
      collection_name: this.fileCollectionName,
      data: [
        {
          id: fullEntry.id,
          text: fullEntry.text,
          vector: fullEntry.vector,
          path: fullEntry.path,
          lineStart: fullEntry.lineStart,
          lineEnd: fullEntry.lineEnd,
          category: fullEntry.category,
          createdAt: fullEntry.createdAt,
          isEvergreen: fullEntry.isEvergreen,
        },
      ],
    });

    await this.client.flush({
      collection_names: [this.fileCollectionName],
    });

    return fullEntry;
  }

  async searchFiles(vector: number[], limit = 5, minScore = 0.5): Promise<MemorySearchResult<FileMemoryEntry>[]> {
    await this.ensureInitialized();

    const results = await this.client.search({
      collection_name: this.fileCollectionName,
      data: [vector],
      limit,
      output_fields: ["text", "path", "lineStart", "lineEnd", "category", "createdAt", "isEvergreen"],
    });

    if (!results || !results.results || results.results.length === 0) {
      return [];
    }

    const mapped = results.results.map((result: any) => ({
      entry: {
        id: result.id,
        text: result.text || "",
        vector: [],
        path: result.path || "",
        lineStart: result.lineStart || 0,
        lineEnd: result.lineEnd || 0,
        category: result.category || "file",
        createdAt: result.createdAt || Date.now(),
        isEvergreen: result.isEvergreen || false,
      },
      score: result.score || 0,
    }));

    return mapped.filter((r) => r.score >= minScore);
  }

  async searchFilesBM25(query: string, limit = 5): Promise<MemorySearchResult<FileMemoryEntry>[]> {
    await this.ensureInitialized();

    try {
      const results = await this.client.search({
        collection_name: this.fileCollectionName,
        data: [query], // Text query for BM25
        limit,
        output_fields: ["text", "path", "lineStart", "lineEnd", "category", "createdAt", "isEvergreen"],
        anns_field: "text", // Search in text field
      });

      if (!results || !results.results || results.results.length === 0) {
        return [];
      }

      const mapped = results.results.map((result: any) => ({
        entry: {
          id: result.id,
          text: result.text || "",
          vector: [],
          path: result.path || "",
          lineStart: result.lineStart || 0,
          lineEnd: result.lineEnd || 0,
          category: result.category || "file",
          createdAt: result.createdAt || Date.now(),
          isEvergreen: result.isEvergreen || false,
        },
        score: result.score || 0,
      }));

      return mapped;
    } catch (error: any) {
      // BM25 might not be available
      console.warn(`BM25 search failed: ${error.message}`);
      return [];
    }
  }

  async deleteFile(id: string): Promise<boolean> {
    await this.ensureInitialized();
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(id)) {
      throw new Error(`Invalid memory ID format: ${id}`);
    }
    await this.client.delete({
      collection_name: this.fileCollectionName,
      expr: `id == "${id}"`,
    });
    return true;
  }

  async countFiles(): Promise<number> {
    await this.ensureInitialized();
    const stats = await this.client.getCollectionStatistics({
      collection_name: this.fileCollectionName,
    });
    const rowCount = stats.data?.row_count;
    if (typeof rowCount === "number") {
      return rowCount;
    }
    if (typeof rowCount === "string") {
      return parseInt(rowCount, 10);
    }
    return 0;
  }

  // ========================================================================
  // Legacy Methods (for backward compatibility)
  // ========================================================================

  async store(entry: Omit<ConversationMemoryEntry, "id" | "createdAt">): Promise<ConversationMemoryEntry> {
    return this.storeConversation(entry);
  }

  async search(vector: number[], limit = 5, minScore = 0.5): Promise<MemorySearchResult<ConversationMemoryEntry>[]> {
    return this.searchConversations(vector, limit, minScore);
  }

  async delete(id: string): Promise<boolean> {
    return this.deleteConversation(id);
  }

  async count(): Promise<number> {
    return this.countConversations();
  }
}
