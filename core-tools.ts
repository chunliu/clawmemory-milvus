/**
 * Core Memory Tools for Milvus
 *
 * Implements memory_search and memory_get tools to replace memory-core.
 * Provides file-backed memory search with hybrid vector + BM25 search.
 */

import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { MemoryDB } from "./milvus-provider.js";
import type { Embeddings } from "./embeddings.js";
import type { CoreMemoryConfig } from "./config.js";

// ============================================================================
// Types
// ============================================================================

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

export interface MemorySearchResult {
  path: string;
  lines: number[];
  text: string;
  score: number;
}

// ============================================================================
// MMR (Maximal Marginal Relevance) Implementation
// ============================================================================

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vector dimensions must match");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

export function applyMMR(
  results: Array<{ entry: FileMemoryEntry; score: number }>,
  queryVector: number[],
  lambda: number = 0.7,
): Array<{ entry: FileMemoryEntry; score: number }> {
  if (results.length === 0) {
    return [];
  }

  const selected: Array<{ entry: FileMemoryEntry; score: number }> = [];
  const remaining = [...results];

  // Select the first result (highest relevance)
  remaining.sort((a, b) => b.score - a.score);
  selected.push(remaining.shift()!);

  // Greedily select remaining results
  while (remaining.length > 0) {
    let bestIndex = -1;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];

      // Calculate relevance to query
      const relevance = candidate.score;

      // Calculate diversity (max similarity to already selected)
      let maxSimilarity = 0;
      for (const selectedEntry of selected) {
        const similarity = cosineSimilarity(
          candidate.entry.vector,
          selectedEntry.entry.vector,
        );
        maxSimilarity = Math.max(maxSimilarity, similarity);
      }

      // MMR score
      const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarity;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIndex = i;
      }
    }

    if (bestIndex >= 0) {
      selected.push(remaining.splice(bestIndex, 1)[0]);
    } else {
      break;
    }
  }

  return selected;
}

// ============================================================================
// Temporal Decay Implementation
// ============================================================================

export function applyTemporalDecay(
  results: Array<{ entry: FileMemoryEntry; score: number }>,
  halfLifeDays: number = 30,
): Array<{ entry: FileMemoryEntry; score: number }> {
  const now = Date.now();
  const halfLifeMs = halfLifeDays * 24 * 60 * 60 * 1000;

  return results.map((result) => {
    const ageMs = now - result.entry.createdAt;
    const decay = Math.pow(0.5, ageMs / halfLifeMs);

    // Boost evergreen content (MEMORY.md, non-date files)
    const evergreenBoost = result.entry.isEvergreen ? 1.2 : 1.0;

    return {
      ...result,
      score: result.score * decay * evergreenBoost,
    };
  });
}

// ============================================================================
// Memory Search Tool
// ============================================================================

export function createMemorySearchTool(
  db: MemoryDB,
  embeddings: Embeddings,
  config: CoreMemoryConfig,
) {
  return {
    name: "memory_search",
    label: "Memory Search",
    description:
      "Mandatory recall step: semantically search MEMORY.md + memory/*.md (and optional session transcripts) before answering questions about prior work, decisions, dates, people, preferences, or todos; returns top snippets with path + lines. If response has disabled=true, memory retrieval is unavailable and should be surfaced to the user.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      maxResults: Type.Optional(
        Type.Number({
          description: "Maximum number of results (default: 5)",
          minimum: 1,
          maximum: 20,
        }),
      ),
      minScore: Type.Optional(
        Type.Number({
          description: "Minimum similarity score (default: 0.3)",
          minimum: 0,
          maximum: 1,
        }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { query, maxResults = 5, minScore = 0.3 } = params as {
        query: string;
        maxResults?: number;
        minScore?: number;
      };

      try {
        // Generate query vector
        const queryVector = await embeddings.embed(query);

        // Search in Milvus
        const results = await db.searchFiles(
          queryVector,
          maxResults * (config.search.hybrid.candidateMultiplier || 4),
          minScore,
        );

        if (results.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: "No relevant memories found in MEMORY.md or memory/*.md files.",
              },
            ],
            details: { disabled: false, count: 0 },
          };
        }

        // Apply MMR if enabled
        let processedResults = results;
        if (config.search.mmr.enabled) {
          processedResults = applyMMR(
            results,
            queryVector,
            config.search.mmr.lambda || 0.7,
          );
        }

        // Apply temporal decay if enabled
        if (config.search.temporalDecay.enabled) {
          processedResults = applyTemporalDecay(
            processedResults,
            config.search.temporalDecay.halfLifeDays || 30,
          );
        }

        // Sort by score and limit
        processedResults.sort((a, b) => b.score - a.score);
        processedResults = processedResults.slice(0, maxResults);

        // Format results
        const formattedResults = processedResults.map((r) => ({
          path: r.entry.path,
          lines: [r.entry.lineStart, r.entry.lineEnd],
          text: r.entry.text,
          score: r.score,
        }));

        const output = formattedResults
          .map(
            (r) =>
              `**${r.path}** (lines ${r.lines[0]}-${r.lines[1]}, score: ${r.score.toFixed(2)})\n${r.text}`,
          )
          .join("\n\n");

        return {
          content: [{ type: "text", text: output }],
          details: { disabled: false, count: formattedResults.length },
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Memory search failed: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          details: { disabled: true, error: String(error) },
        };
      }
  },
  };
}

// ============================================================================
// Memory Get Tool
// ============================================================================

export function createMemoryGetTool(api: OpenClawPluginApi) {
  return {
    name: "memory_get",
    label: "Memory Get",
    description: "Read content from MEMORY.md or memory/*.md files. Use for retrieving specific file contents.",
    parameters: Type.Object({
      path: Type.String({ description: "File path (e.g., MEMORY.md or memory/2026-03-07.md)" }),
      from: Type.Optional(
        Type.Number({
          description: "Starting line number (1-indexed)",
          minimum: 1,
        }),
      ),
      lines: Type.Optional(
        Type.Number({
          description: "Number of lines to read",
          minimum: 1,
        }),
      ),
    }),
    async execute(_toolCallId: string, params: unknown) {
      const { path, from, lines } = params as {
        path: string;
        from?: number;
        lines?: number;
      };

      try {
        // Validate path
        const allowedPaths = ["MEMORY.md", "memory"];
        const isAllowed =
          path === "MEMORY.md" ||
          path.startsWith("memory/") ||
          allowedPaths.some((p) => path.startsWith(p));

        if (!isAllowed) {
          return {
            content: [
              {
                type: "text",
                text: `Access denied: path "${path}" is not allowed. Only MEMORY.md and memory/*.md files are accessible.`,
              },
            ],
            details: { error: "access_denied", path },
          };
        }

        // Resolve full path (relative to workspace)
        const workspace = process.env.HOME
          ? `${process.env.HOME}/.openclaw/workspace`
          : "/tmp/openclaw-workspace";
        const fullPath = `${workspace}/${path}`;

        // Read file
        const fs = await import("node:fs/promises");
        const content = await fs.readFile(fullPath, "utf-8");

        // Extract line range if specified
        if (from !== undefined || lines !== undefined) {
          const allLines = content.split("\n");
          const startLine = from !== undefined ? from - 1 : 0; // Convert to 0-indexed
          const numLines = lines !== undefined ? lines : allLines.length - startLine;
          const endLine = Math.min(startLine + numLines, allLines.length);

          const extractedLines = allLines.slice(startLine, endLine);
          const extractedContent = extractedLines.join("\n");

          return {
            content: [{ type: "text", text: extractedContent }],
            details: {
              path,
              from,
              lines: extractedLines.length,
              totalLines: allLines.length,
            },
          };
        }

        return {
          content: [{ type: "text", text: content }],
          details: { path, lines: content.split("\n").length },
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Failed to read file: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          details: {
            error: String(error),
            path,
          },
        };
      }
    },
  };
}
