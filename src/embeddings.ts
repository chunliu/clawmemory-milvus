/**
 * Embedding provider support
 */

import crypto from "node:crypto";
import type { EmbeddingProvider, EmbeddingProviderResult } from "./types.js";

export interface EmbeddingsConfig {
  provider: EmbeddingProvider;
  model: string;
  apiKey?: string;
  baseUrl?: string;
  dimensions?: number;
}

export class EmbeddingsProvider {
  private provider: EmbeddingProvider;
  private model: string;
  private apiKey?: string;
  private baseUrl?: string;
  private dimensions?: number;
  private cache: Map<string, number[]>;
  private cacheMaxEntries: number;

  constructor(config: EmbeddingsConfig & { cacheMaxEntries?: number }) {
    this.provider = config.provider;
    this.model = config.model;
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl;
    this.dimensions = config.dimensions;
    this.cache = new Map();
    this.cacheMaxEntries = config.cacheMaxEntries || 50000;
  }

  async embed(text: string): Promise<number[]> {
    // Check cache
    const cacheKey = this.getCacheKey(text);
    const cached = this.cache.get(cacheKey);
    if (cached) {
      return cached;
    }

    // Generate embedding
    const embedding = await this.generateEmbedding(text);

    // Cache result
    this.cache.set(cacheKey, embedding);

    // Evict old entries if cache is too large
    if (this.cache.size > this.cacheMaxEntries) {
      const keysToDelete = Array.from(this.cache.keys()).slice(
        0,
        this.cache.size - this.cacheMaxEntries,
      );
      for (const key of keysToDelete) {
        this.cache.delete(key);
      }
    }

    return embedding;
  }

  private getCacheKey(text: string): string {
    return crypto.createHash("sha256").update(text).digest("hex");
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    switch (this.provider) {
      case "openai":
        return this.embedOpenAI(text);
      case "gemini":
        return this.embedGemini(text);
      case "voyage":
        return this.embedVoyage(text);
      case "mistral":
        return this.embedMistral(text);
      case "ollama":
        return this.embedOllama(text);
      case "auto":
        return this.embedAuto(text);
      default:
        throw new Error(`Unsupported embedding provider: ${this.provider}`);
    }
  }

  private async embedOpenAI(text: string): Promise<number[]> {
    // Import OpenAI SDK dynamically
    const { default: OpenAI } = await import("openai");

    const client = new OpenAI({
      apiKey: this.apiKey || process.env.OPENAI_API_KEY,
      baseURL: this.baseUrl,
    });

    const params: any = {
      model: this.model,
      input: text,
    };

    if (this.dimensions) {
      params.dimensions = this.dimensions;
    }

    const response = await client.embeddings.create(params);
    return response.data[0].embedding;
  }

  private async embedGemini(text: string): Promise<number[]> {
    // Import Google Generative AI SDK
    const { GoogleGenerativeAI } = await import("@google/generative-ai");

    const genAI = new GoogleGenerativeAI(
      this.apiKey || process.env.GOOGLE_API_KEY || "",
    );
    const model = genAI.getGenerativeModel({ model: this.model });

    const result = await model.embedContent(text);
    return result.embedding.values;
  }

  private async embedVoyage(text: string): Promise<number[]> {
    // Import Voyage AI SDK
    const Voyage = await import("voyageai");

    const client = new Voyage.VoyageAI({
      apiKey: this.apiKey || process.env.VOYAGE_API_KEY,
    });

    const response = await client.embed({
      inputs: [text],
      model: this.model,
      inputType: "document",
    });

    return response.data[0].embedding;
  }

  private async embedMistral(text: string): Promise<number[]> {
    // Import Mistral AI SDK
    const Mistral = await import("@mistralai/mistralai");

    const client = new Mistral.MistralClient({
      apiKey: this.apiKey || process.env.MISTRAL_API_KEY,
    });

    const response = await client.embeddings.create({
      model: this.model,
      inputs: [text],
    });

    return response.data[0].embedding;
  }

  private async embedOllama(text: string): Promise<number[]> {
    // Use fetch for Ollama
    const baseUrl = this.baseUrl || "http://localhost:11434";

    const response = await fetch(`${baseUrl}/api/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embedding failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  }

  private async embedAuto(text: string): Promise<number[]> {
    // Auto-detect provider based on model name
    if (this.model.startsWith("text-embedding")) {
      return this.embedOpenAI(text);
    } else if (this.model.startsWith("models/")) {
      return this.embedGemini(text);
    } else if (this.model.startsWith("voyage-")) {
      return this.embedVoyage(text);
    } else if (this.model.startsWith("mistral-")) {
      return this.embedMistral(text);
    } else {
      // Default to OpenAI
      return this.embedOpenAI(text);
    }
  }

  getProviderInfo(): EmbeddingProviderResult {
    return {
      provider: this.provider,
      model: this.model,
      requestedProvider: this.provider,
    };
  }

  clearCache(): void {
    this.cache.clear();
  }

  getCacheSize(): number {
    return this.cache.size;
  }
}

export async function createEmbeddingsProvider(
  config: EmbeddingsConfig & { cacheMaxEntries?: number },
): Promise<EmbeddingsProvider> {
  return new EmbeddingsProvider(config);
}
