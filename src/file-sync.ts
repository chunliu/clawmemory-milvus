/**
 * File synchronization: read memory files and sync to Milvus
 */

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import type { Chunk, MemorySource } from "./types.js";

export interface FileSyncConfig {
  workspaceDir: string;
  sources: Array<"memory" | "sessions">;
  chunkSize: number;
  chunkOverlap: number;
}

export class FileSync {
  private config: FileSyncConfig;

  constructor(config: FileSyncConfig) {
    this.config = config;
  }

  async sync(progress?: (update: { completed: number; total: number; label?: string }) => void): Promise<Chunk[]> {
    const chunks: Chunk[] = [];
    const files = await this.getMemoryFiles();

    if (progress) {
      progress({ completed: 0, total: files.length, label: "Reading files" });
    }

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const fileChunks = await this.processFile(file);

      chunks.push(...fileChunks);

      if (progress) {
        progress({
          completed: i + 1,
          total: files.length,
          label: `Processing ${file.path}`,
        });
      }
    }

    return chunks;
  }

  private async getMemoryFiles(): Promise<Array<{ path: string; source: MemorySource }>> {
    const files: Array<{ path: string; source: MemorySource }> = [];
    const { workspaceDir, sources } = this.config;

    if (sources.includes("memory")) {
      // Add MEMORY.md
      const memoryMdPath = path.join(workspaceDir, "MEMORY.md");
      try {
        await fs.access.stat(memoryMdPath);
        files.push({ path: "MEMORY.md", source: "memory" });
      } catch {
        // File doesn't exist
      }

      // Add memory/ directory
      const memoryDir = path.join(workspaceDir, "memory");
      try {
        const entries = await fs.readdir(memoryDir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isFile() && entry.name.endsWith(".md")) {
            files.push({ path: `memory/${entry.name}`, source: "memory" });
          }
        }
      } catch {
        // Directory doesn't exist
      }
    }

    // TODO: Add sessions source if needed

    return files;
  }

  private async processFile(file: { path: string; source: MemorySource }): Promise<Chunk[]> {
    const fullPath = path.join(this.config.workspaceDir, file.path);

    try {
      const content = await fs.readFile(fullPath, "utf-8");
      const lines = content.split("\n");

      return this.chunkText(lines, file.path, file.source);
    } catch (error) {
      console.error(`Failed to read file ${file.path}:`, error);
      return [];
    }
  }

  private chunkText(
    lines: string[],
    filePath: string,
    source: MemorySource,
  ): Chunk[] {
    const chunks: Chunk[] = [];
    const { chunkSize, chunkOverlap } = this.config;

    for (let i = 0; i < lines.length; i += chunkSize - chunkOverlap) {
      const startLine = i + 1;
      const endLine = Math.min(i + chunkSize, lines.length);
      const text = lines.slice(i, endLine).join("\n");

      if (text.trim().length === 0) {
        continue;
      }

      chunks.push({
        id: this.generateChunkId(filePath, startLine, endLine),
        path: filePath,
        startLine,
        endLine,
        text,
        embedding: [], // Will be filled by embedding provider
        model: "", // Will be filled by embedding provider
        source,
      });
    }

    return chunks;
  }

  private generateChunkId(filePath: string, startLine: number, endLine: number): string {
    const hash = crypto
      .createHash("sha256")
      .update(`${filePath}:${startLine}:${endLine}`)
      .digest("hex");
    return hash.slice(0, 16);
  }
}
