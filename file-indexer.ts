/**
 * File Indexer
 *
 * Watches and indexes MEMORY.md and memory/*.md files into Milvus.
 * Chunks files, generates embeddings, and stores in file collection.
 */

import { promises as fs } from "node:fs";
import { glob } from "glob";
import type { MemoryDB, FileMemoryEntry } from "./milvus-provider.js";
import type { Embeddings } from "./embeddings.js";
import type { CoreMemoryConfig } from "./config.js";

// ============================================================================
// Types
// ============================================================================

export interface FileChunk {
  id: string;
  text: string;
  path: string;
  lineStart: number;
  lineEnd: number;
  isEvergreen: boolean;
}

// ============================================================================
// File Chunking
// ============================================================================

/**
 * Approximate token count (roughly 4 chars per token)
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Chunk a file into overlapping pieces
 */
export function chunkFile(
  content: string,
  path: string,
  chunkSize: number = 400,
  chunkOverlap: number = 80,
): FileChunk[] {
  const lines = content.split("\n");
  const chunks: FileChunk[] = [];

  let currentChunk = "";
  let lineStart = 1;
  let currentTokens = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineTokens = estimateTokens(line + "\n");

    // If adding this line would exceed chunk size, save current chunk
    if (currentTokens + lineTokens > chunkSize && currentChunk.length > 0) {
      chunks.push({
        id: `${path}:${lineStart}-${i}`,
        text: currentChunk.trim(),
        path,
        lineStart,
        lineEnd: i,
        isEvergreen: isEvergreenFile(path),
      });

      // Start new chunk with overlap
      const overlapLines = Math.floor(chunkOverlap / 4); // Approximate lines
      const overlapStart = Math.max(0, i - overlapLines);
      currentChunk = lines.slice(overlapStart, i + 1).join("\n");
      lineStart = overlapStart + 1;
      currentTokens = estimateTokens(currentChunk);
    } else {
      currentChunk += (currentChunk ? "\n" : "") + line;
      currentTokens += lineTokens;
    }
  }

  // Add final chunk if non-empty
  if (currentChunk.trim().length > 0) {
    chunks.push({
      id: `${path}:${lineStart}-${lines.length}`,
      text: currentChunk.trim(),
      path,
      lineStart,
      lineEnd: lines.length,
      isEvergreen: isEvergreenFile(path),
    });
  }

  return chunks;
}

/**
 * Check if a file is evergreen (MEMORY.md or non-date file)
 */
function isEvergreenFile(path: string): boolean {
  // MEMORY.md is always evergreen
  if (path === "MEMORY.md") {
    return true;
  }

  // Files in memory/ directory
  if (path.startsWith("memory/")) {
    const filename = path.replace("memory/", "");

    // Non-date files are evergreen
    if (!/^\d{4}-\d{2}-\d{2}\.md$/.test(filename)) {
      return true;
    }
  }

  return false;
}

// ============================================================================
// File Indexer Class
// ============================================================================

export class FileIndexer {
  private watchPaths: string[];
  private chunkSize: number;
  private chunkOverlap: number;
  private indexing = false;
  private indexQueue = new Set<string>();

  constructor(
    private db: MemoryDB,
    private embeddings: Embeddings,
    config: CoreMemoryConfig,
    private logger: any,
    private agentId?: string,
  ) {
    this.watchPaths = config.watchPaths || ["MEMORY.md", "memory/"];
    this.chunkSize = config.chunkSize || 400;
    this.chunkOverlap = config.chunkOverlap || 80;
  }

  /**
   * Initialize indexer (index all files)
   */
  async initialize(workspace?: string): Promise<void> {
    this.logger.info("file-indexer: initializing");

    // Get workspace path (agent-specific if provided)
    const resolvedWorkspace = workspace || (process.env.HOME
      ? `${process.env.HOME}/.openclaw/workspace`
      : "/tmp/openclaw-workspace");

    // Find all files to index
    const filesToIndex = await this.findFilesToIndex(resolvedWorkspace);
    this.logger.info(`file-indexer: found ${filesToIndex.length} files to index`);

    // Index each file
    for (const file of filesToIndex) {
      await this.indexFile(file, resolvedWorkspace);
    }

    this.logger.info("file-indexer: initialization complete");
  }

  /**
   * Find all files that should be indexed
   */
  private async findFilesToIndex(workspace: string): Promise<string[]> {
    const files: string[] = [];

    for (const watchPath of this.watchPaths) {
      const fullPath = `${workspace}/${watchPath}`;

      try {
        const stats = await fs.stat(fullPath);

        if (stats.isFile()) {
          files.push(watchPath);
        } else if (stats.isDirectory()) {
          // Find all .md files in directory
          const pattern = `${fullPath}/**/*.md`;
          const matched = await glob(pattern);
          files.push(...matched.map((f) => f.replace(workspace + "/", "")));
        }
      } catch (error) {
        this.logger.warn(`file-indexer: cannot access ${watchPath}: ${error}`);
      }
    }

    return files;
  }

  /**
   * Index a single file
   */
  async indexFile(path: string, workspace?: string): Promise<void> {
    // Get workspace path (agent-specific if provided)
    const resolvedWorkspace = workspace || (process.env.HOME
      ? `${process.env.HOME}/.openclaw/workspace`
      : "/tmp/openclaw-workspace");
    const fullPath = `${resolvedWorkspace}/${path}`;

    try {
      // Read file content
      const content = await fs.readFile(fullPath, "utf-8");

      // Chunk file
      const chunks = chunkFile(content, path, this.chunkSize, this.chunkOverlap);
      this.logger.info(`file-indexer: chunked ${path} into ${chunks.length} chunks`);

      // Generate embeddings for all chunks
      const texts = chunks.map((c) => c.text);
      const vectors = await this.embeddings.embedBatch(texts);

      // Store chunks in Milvus
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const vector = vectors[i];

        await this.db.storeFile({
          text: chunk.text,
          vector,
          path: chunk.path,
          lineStart: chunk.lineStart,
          lineEnd: chunk.lineEnd,
          category: "file",
          isEvergreen: chunk.isEvergreen,
          agentId: this.agentId,
        });
      }

      this.logger.info(`file-indexer: indexed ${path} (${chunks.length} chunks)`);
    } catch (error) {
      this.logger.error(`file-indexer: failed to index ${path}: ${error}`);
    }
  }

  /**
   * Queue a file for re-indexing
   */
  queueReindex(path: string): void {
    this.indexQueue.add(path);
    this.processQueue();
  }

  /**
   * Process the index queue
   */
  private async processQueue(): Promise<void> {
    if (this.indexing || this.indexQueue.size === 0) {
      return;
    }

    this.indexing = true;

    try {
      for (const path of this.indexQueue) {
        await this.indexFile(path);
        this.indexQueue.delete(path);
);
      }
    } finally {
      this.indexing = false;
    }
  }

  /**
   * Start watching files for changes
   */
  async startWatching(): Promise<void> {
    this.logger.info("file-indexer: starting file watcher");

    // For now, just log that watching is enabled
    // In a full implementation, we would use chokidar or similar
    // to watch files and trigger re-indexing on changes

    this.logger.info("file-indexer: file watcher started (note: auto-reindex not yet implemented)");
  }
}
