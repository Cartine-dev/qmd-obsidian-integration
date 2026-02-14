/**
 * openai_compatible.ts - OpenAI-compatible embeddings provider for QMD
 *
 * Provides embeddings via HTTP requests to OpenAI-compatible endpoints
 * (e.g., LiteLLM, Infinity, OpenAI API).
 *
 * Configuration via environment variables:
 * - QMD_OPENAI_BASE_URL: Base URL for the API (e.g., http://localhost:4000)
 * - QMD_OPENAI_API_KEY: API key for authentication
 * - QMD_OPENAI_EMBEDDING_MODEL: Model name (default: default-embedding-model)
 * - QMD_OPENAI_TIMEOUT_MS: Request timeout in ms (default: 30000)
 */

import type { EmbeddingResult } from "../llm";

/**
 * Configuration for OpenAI-compatible provider
 */
export type OpenAIConfig = {
    baseUrl: string;
    apiKey: string;
    model: string;
    timeoutMs: number;
};

/**
 * OpenAI API embeddings request payload
 */
type EmbeddingsRequest = {
    model: string;
    input: string | string[];
    encoding_format?: "float" | "base64";
};

/**
 * OpenAI API embeddings response
 */
type EmbeddingsResponse = {
    object: "list";
    data: Array<{
        object: "embedding";
        index: number;
        embedding: number[];
    }>;
    model: string;
    usage: {
        prompt_tokens: number;
        total_tokens: number;
    };
};

/**
 * Load configuration from environment variables
 */
export function loadOpenAIConfig(): OpenAIConfig | null {
    const baseUrl = Bun.env.QMD_OPENAI_BASE_URL;
    const apiKey = Bun.env.QMD_OPENAI_API_KEY;

    if (!baseUrl || !apiKey) {
        return null;
    }

    return {
        baseUrl: baseUrl.replace(/\/$/, ""), // Remove trailing slash
        apiKey,
        model: Bun.env.QMD_OPENAI_EMBEDDING_MODEL || "default-embedding-model",
        timeoutMs: parseInt(Bun.env.QMD_OPENAI_TIMEOUT_MS || "30000", 10),
    };
}

/**
 * OpenAI-compatible embeddings provider
 */
export class OpenAICompatibleEmbeddings {
    private config: OpenAIConfig;

    constructor(config: OpenAIConfig) {
        this.config = config;
    }

    /**
     * Embed a single text
     */
    async embed(text: string): Promise<EmbeddingResult | null> {
        const results = await this.embedBatch([text]);
        return results[0] ?? null;
    }

    /**
     * Embed multiple texts in a single batch request
     */
    async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
        if (texts.length === 0) {
            return [];
        }

        const url = `${this.config.baseUrl}/v1/embeddings`;
        const payload: EmbeddingsRequest = {
            model: this.config.model,
            input: texts,
            encoding_format: "float",
        };

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${this.config.apiKey}`,
                },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`OpenAI API error (${response.status}): ${errorText}`);
                return texts.map(() => null);
            }

            const data = await response.json() as EmbeddingsResponse;

            // Map response to EmbeddingResult format
            // Sort by index to ensure correct order
            const sorted = data.data.sort((a, b) => a.index - b.index);
            return sorted.map(item => ({
                embedding: item.embedding,
                model: data.model,
            }));

        } catch (err) {
            if (err instanceof Error) {
                if (err.name === "AbortError") {
                    console.error(`OpenAI API timeout after ${this.config.timeoutMs}ms`);
                } else {
                    console.error(`OpenAI API request failed: ${err.message}`);
                }
            }
            return texts.map(() => null);
        }
    }
}
