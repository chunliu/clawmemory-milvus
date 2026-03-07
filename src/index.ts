/**
 * OpenClaw Memory (Milvus) Plugin
 *
 * Milvus-backed memory with hybrid search (vector + BM25).
 * Provides memory_search and memory_get tools compatible with OpenClaw.
 */

import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/memory-milvus";
import { milvusConfigSchema, type MilvusConfig } from "./config.js";
import { MilvusMemoryManager } from "./milvus-client.js";
import { EmbeddingsProvider, createEmbeddingsProvider } from "./embeddings.js";
import { applyMMR } from "./mmr.js";
import { applyTemporalDecayToHybridResults } from "./temporal-decay.js";
import { FileWatcher } from "./file-watcher.js";
import { FileSync } from "./file-sync.js";
import type { MemorySearchResult, MemorySource } from "./types.js";
import fs from "node:fs/promises";
import path from "node:path";

const memoryMilvusPlugin = {
  id: "memory-milvus",
  name: "Memory (Milvus)",
  description: "Milvus-backed memory with hybrid search (vector + BM25)",
  kind: "memory" as const,
  configSchema: milvusConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = milvusConfigSchema.parse(api.pluginConfig);

    // Resolve workspace directory
    const workspaceDir = api.resolvePath(".");

    // Get agent ID from session key (OpenClaw provides this)
    // Format: "agent:{agentId}" or similar
    const agentId = api.sessionKey?.split(":")[1] || "default";

    // Initialize Milvus client with agent-specific collection
    const manager = new MilvusMemoryManager({
      address: cfg.milvus.address,
      baseCollectionName: cfg.milvus.collectionName,
      agentId,
      vectorDim: cfg.embedding.dimensions || 1536, // Default for text-embedding-3-small
      username: cfg.milvus.username,
      password: cfg.milvus.password,
    });

    // Initialize embeddings provider
    const embeddings = new EmbeddingsProvider({
      provider: cfg.embedding.provider,
      model: cfg.embedding.model,
      apiKey: cfg.embedding.apiKey,
      baseUrl: cfg.embedding.baseUrl,
      dimensions: cfg.embedding.dimensions,
      cacheMaxEntries: cfg.cache.maxEntries,
    });

    // Initialize file sync
    const fileSync = new FileSync({
      workspaceDir,
      sources: cfg.sources || ["memory"],
      chunkSize: 50,
      chunkOverlap: 5,
    });

    // Initialize file watcher
    const fileWatcher = new FileWatcher({
      workspaceDir,
      sources: cfg.sources || ["memory"],
      debounceMs: cfg.sync.debounceMs,
      onChange: () => {
        manager.markDirty();
        if (cfg.sync.onSearch) {
          // Sync will be triggered on next search
        }
      },
    });

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: "memory_search",
        label: "Memory Search",
        description:
          "Search memory files using hybrid vector + BM25 search. Returns top snippets with path + lines.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          maxResults: Type.Optional(
            Type.Number({ description: "Max results (default: 5)" }),
          ),
          minScore: Type.Optional(
            Type.Number({ description: "Minimum similarity score (0-1)" }),
          ),
        }),
        execute: async (_toolCallId, params) => {
          const { query, maxResults, minScore } = params as {
            query: string;
            maxResults?: number;
            minScore?: number;
          };

          // Sync if dirty
          if (cfg.sync.onSearch && manager.isDirty()) {
            await performSync();
          }

          // Embed query
          const queryVector = await embeddings.embed(query);

          let results: any[];

          if (cfg.search.hybrid.enabled) {
            // Hybrid search
            results = await manager.hybridSearch({
              queryVector,
              queryText: query,
              vectorWeight: cfg.search.hybrid.vectorWeight,
              textWeight: cfg.search.hybrid.textWeight,
              limit: maxResults || cfg.search.maxResults,
              model: embeddings.getProviderInfo().model,
            });
          } else {
            // Vector-only search
            results = await manager.searchVector(
              queryVector,
              maxResults || cfg.search.maxResults,
              embeddings.getProviderInfo().model,
            );
          }

          // Apply MMR
          if (cfg.search.mmr.enabled) {
            results = applyMMR(results, {
              enabled: true,
              lambda: cfg.search.mmr.lambda,
            });
          }

          // Apply temporal decay
          if (cfg.search.temporalDecay.enabled) {
            results = await applyTemporalDecayToHybridResults({
              results,
              temporalDecay: {
                enabled: true,
                halfLifeDays: cfg.search.temporalDecay.halfLifeDays,
              },
              workspaceDir,
            });
          }

          // Filter by min score
          const min = minScore ?? cfg.search.minScore;
          results = results.filter((r) => r.score >= min);

          // Format results
          const formatted: MemorySearchResult[] = results.map((r) => ({
            path: r.path,
            startLine: r.startLine,
            endLine: r.endLine,
            score: r.score,
            snippet: r.text.slice(0, 700),
            source: r.source as MemorySource,
          }));

          return {
            results: formatted,
            provider: embeddings.getProviderInfo().provider,
            model: embeddings.getProviderInfo().model,
            hybrid: cfg.search.hybrid.enabled,
          };
        },
      },
      { name: "memory_search" },
    );

    api.registerTool(
      {
        name: "memory_get",
        label: "Memory Get",
        description:
          "Read memory file content with optional from/lines. Use after memory_search to pull specific lines.",
        parameters: Type.Object({
          path: Type.String({ description: "File path relative to workspace" }),
          from: Type.Optional(Type.Number({ description: "Start line (1-indexed)" })),
          lines: Type.Optional(Type.Number({ description: "Number of lines to read" })),
        }),
        execute: async (_toolCallId, params) => {
          const { path: relPath, from, lines } = params as {
            path: string;
            from?: number;
            lines?: number;
          };

          // Validate path
          const normalizedPath = relPath.replace(/\\/g, "/");
          if (
            !normalizedPath.startsWith("memory/") &&
            normalizedPath !== "MEMORY.md" &&
            normalizedPath !== "memory.md"
          ) {
            return {
              text: "",
              path: relPath,
              error: "Invalid path: must be MEMORY.md or memory/*.md",
            };
          }

          const fullPath = path.join(workspaceDir, normalizedPath);

          try {
            const content = await fs.readFile(fullPath, "utf-8");
            const allLines = content.split("\n");

            const startLine = from ? Math.max(1, from) : 1;
            const lineCount = lines || allLines.length - startLine + 1;
            const endLine = Math.min(startLine + lineCount - 1, allLines.length);

            const selectedLines = allLines.slice(startLine - 1, endLine);
            const text = selectedLines.join("\n");

            return {
              text,
              path: normalizedPath,
            };
          } catch (error) {
            return {
              text: "",
              path: relPath,
              error: error instanceof Error ? error.message : String(error),
            };
          }
        },
      },
      { name: "memory_get" },
    );

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const memory = program
          .command("memory")
          .description("Milvus memory plugin commands");

        memory
          .command("status")
          .description("Show memory status")
          .action(async () => {
            const status = manager.getStatus();
            console.log(`Agent ID: ${agentId}`);
            console.log(`Collection: ${manager.getCollectionName()}`);
            console.log(JSON.stringify(status, null, 2));
          });

        memory
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "5")
          .action(async (query, opts) => {
            const queryVector = await embeddings.embed(query);
            const results = await manager.searchVector(
              queryVector,
              parseInt(opts.limit),
            );
            console.log(JSON.stringify(results, null, 2));
          });

        memory
          .command("sync")
          .description("Sync memory files to Milvus")
          .action(async () => {
            await performSync();
            console.log("Sync completed");
          });
      },
      { commands: ["memory"] },
    );

    // ========================================================================
    // Service
    // ========================================================================

    async function performSync(): Promise<void> {
      if (manager.isDirty() || !manager.isDirty()) {
        // Always sync
      }

      const chunks = await fileSync.sync((update) => {
        api.logger.info?.(
          `Syncing memory: ${update.completed}/${update.total} - ${update.label || ""}`,
        );
      });

      // Generate embeddings for all chunks
      for (const chunk of chunks) {
        chunk.embedding = await embeddings.embed(chunk.text);
        chunk.model = embeddings.getProviderInfo().model;
      }

      // Insert into Milvus
      await manager.insert(chunks);
      manager.clearDirty();

      api.logger.info?.(`Synced ${chunks.length} chunks to Milvus`);
    }

    api.registerService({
      id: "memory-milvus",
      start: async () => {
        api.logger.info?.(
          `memory-milvus: initializing (address: ${cfg.milvus.address}, collection: ${cfg.milvus.collectionName})`,
        );

        // Initialize Milvus
        await manager.initialize();

        // Start file watcher
        if (cfg.sync.watch) {
          fileWatcher.start();
          api.logger.info?.("memory-milvus: file watcher started");
        }

        // Initial sync
        if (cfg.sync.onSessionStart) {
          await performSync();
        }

        api.logger.info?.("memory-milvus: ready");
      },
      stop: async () => {
        api.logger.info?.("memory-milvus: stopping");

        // Stop file watcher
        fileWatcher.stop();

        // Close Milvus connection
        await manager.close();

        api.logger.info?.("memory-milvus: stopped");
      },
    });
  },
};

export default memoryMilvusPlugin;
