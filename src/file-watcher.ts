/**
 * File watcher for memory files
 */

import chokidar from "chokidar";
import type { Chunk } from "./types.js";

export interface FileWatcherConfig {
  workspaceDir: string;
  sources: Array<"memory" | "sessions">;
  debounceMs: number;
  onChange: () => void;
}

export class FileWatcher {
  private watcher: chokidar.FSWatcher | null = null;
  private debounceTimer: NodeJS.Timeout | null = null;
  private config: FileWatcherConfig;

  constructor(config: FileWatcherConfig) {
    this.config = config;
  }

  start(): void {
    if (this.watcher) {
      return;
    }

    const paths = this.getWatchPaths();

    this.watcher = chokidar.watch(paths, {
      ignored: /(^|[\/\\])\./, // Ignore hidden files
      persistent: true,
      ignoreInitial: true,
    });

    this.watcher.on("all", () => {
      this.scheduleDebounce();
    });
  }

  stop(): void {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }

    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }

  private getWatchPaths(): string[] {
    const paths: string[] = [];
    const { workspaceDir, sources } = this.config;

    if (sources.includes("memory")) {
      paths.push(
        `${workspaceDir}/MEMORY.md`,
        `${workspaceDir}/memory`,
      );
    }

    if (sources.includes("sessions")) {
      // Sessions are typically stored elsewhere
      // This would be configured separately
    }

    return paths;
  }

  private scheduleDebounce(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.config.onChange();
      this.debounceTimer = null;
    }, this.config.debounceMs);
  }
}
