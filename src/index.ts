/**
 * OpenClaw Milvus Memory Backend
 * 
 * This plugin provides a Milvus vector database backend for OpenClaw's
 * memory system, enabling large-scale semantic search over Markdown files.
 */

export { MilvusMemoryBackend } from './milvus-backend';
export * from './types';

/**
 * Plugin factory function for OpenClaw
 */
export function createMilvusBackend(config: any) {
  const { MilvusMemoryBackend } = require('./milvus-backend');
  return new MilvusMemoryBackend(config);
}

// Default export for CommonJS compatibility
module.exports = {
  MilvusMemoryBackend: require('./milvus-backend').MilvusMemoryBackend,
  createMilvusBackend,
};
