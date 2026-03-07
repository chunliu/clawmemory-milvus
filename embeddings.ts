/**
 * Embeddings Provider
 *
 * Supports OpenAI, Ollama, and custom embedding providers.
 */

import OpenAI from "openai";
import type { EmbeddingProvider } from "./config.js";

export class Embeddings {
  private client: OpenAI;
  private provider: EmbeddingProvider;

  constructor(
    provider: EmbeddingProvider,
    apiKey: string | undefined,
    private model: string,
    baseUrl: string,
    private dimensions?: number,
  ) {
    this.provider = provider;
    // For Ollama and custom, apiKey can be "dummy" or undefined
    const effectiveApiKey = apiKey || "dummy";
    this.client = new OpenAI({ apiKey: effectiveApiKey, baseURL: baseUrl });
  }

  async embed(text: string): Promise<number[]> {
    const params: { model: string; input: string; dimensions?: number } = {
      model: this.model,
      input: text,
    };

    // Only add dimensions parameter for OpenAI
    if (this.provider === "openai" && this.dimensions) {
      params.dimensions = this.dimensions;
    }

    const response = await this.client.embeddings.create(params);
    return response.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const params: { model: string; input: string[]; dimensions?: number } = {
      model: this.model,
      input: texts,
    };

    // Only add dimensions parameter for OpenAI
    if (this.provider === "openai" && this.dimensions) {
      params.dimensions = this.dimensions;
    }

    const response = await this.client.embeddings.create(params);
    return response.data.map((item) => item.embedding);
  }
}
