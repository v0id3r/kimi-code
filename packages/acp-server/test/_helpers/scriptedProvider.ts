import {
  IProtocolAdapterRegistry,
  type IProtocolAdapterRegistry as IProtocolAdapterRegistryType,
  type Model,
  ProtocolAdapterRegistry,
} from '@moonshot-ai/agent-core-v2';
import type { Message, StreamedMessagePart } from '@moonshot-ai/agent-core-v2/human/llm/message';
import type { FinishReason } from '@moonshot-ai/agent-core-v2/human/llm-requester/finish-reason';
import type { LlmRequester } from '@moonshot-ai/agent-core-v2/human/llm-requester/requester';
import { coreRequesterFromA } from '../../../agent-core-v2/test/harness/scripted-generate.js';

interface ScriptedResponse {
  readonly parts: readonly StreamedMessagePart[];
  readonly finishReason?: FinishReason | null;
  readonly rawFinishReason?: string | null;
  readonly usage?: TokenUsage;
  readonly gate?: Promise<void>;
}

const ZERO_USAGE: TokenUsage = {
  inputOther: 0,
  output: 0,
  inputCacheRead: 0,
  inputCacheCreation: 0,
};

class ScriptedStream {
  id: string | null = null;
  usage: TokenUsage | null = null;
  finishReason: FinishReason | null = null;
  rawFinishReason: string | null = null;

  constructor(
    private readonly parts: readonly StreamedMessagePart[],
    private readonly response: ScriptedResponse,
    private readonly index: number,
  ) {}

  async *[Symbol.asyncIterator](): AsyncIterator<StreamedMessagePart> {
    if (this.response.gate !== undefined) await this.response.gate;
    for (const part of this.parts) {
      yield part;
    }
    const hasToolCall = this.parts.some((p) => p.type === 'function');
    this.id = `scripted-${String(this.index)}`;
    this.usage = this.response.usage ?? { ...ZERO_USAGE, output: this.parts.length };
    this.finishReason =
      this.response.finishReason ?? (hasToolCall ? 'tool_calls' : 'completed');
    this.rawFinishReason =
      this.response.rawFinishReason ?? (this.finishReason === 'completed' ? 'stop' : this.finishReason);
  }
}

class ScriptedChatProvider {
  readonly name = 'scripted';
  readonly modelName = 'scripted';
  readonly thinkingEffort = null;

  constructor(
    private readonly queue: ScriptedResponse[],
    private readonly calls: Array<readonly Message[]>,
  ) {}

  async generate(
    _systemPrompt: string,
    _tools: readonly Tool[],
    history: readonly Message[],
    options?: { signal?: AbortSignal },
  ): Promise<ScriptedStream> {
    options?.signal?.throwIfAborted();
    const response = this.queue.shift();
    if (response === undefined) {
      throw new Error(
        `scriptedProvider: unexpected generate() call #${String(this.calls.length + 1)} — ` +
          `queue exhausted. Push another response via mockNextResponse().`,
      );
    }
    this.calls.push(history);
    return new ScriptedStream(response.parts, response, this.calls.length);
  }

  withThinking(): ScriptedChatProvider {
    return this;
  }

  withMaxCompletionTokens(): ScriptedChatProvider {
    return this;
  }
}

export interface ScriptedProvider {
  readonly seed: readonly [typeof IProtocolAdapterRegistry, IProtocolAdapterRegistryType];
  mockNextText(text: string): void;
  mockNextTextWithUsage(text: string, usage: TokenUsage): void;
  mockNextGatedText(text: string, gate: Promise<void>): void;
  mockNextResponse(...parts: StreamedMessagePart[]): void;
  mockNextProviderResponse(response: {
    readonly parts?: readonly StreamedMessagePart[];
    readonly finishReason?: FinishReason | null;
    readonly rawFinishReason?: string | null;
  }): void;
  callCount(): number;
  callHistory(): ReadonlyArray<readonly Message[]>;
  systemPrompts(): readonly string[];
}

export function createScriptedProvider(): ScriptedProvider {
  const queue: MockProviderResponse[] = [];
  const calls: Array<readonly Message[]> = [];
  const systemPrompts: string[] = [];
  const provider = new ScriptedChatProvider(queue, calls);
  const requester: LlmRequester = {
    async generate(config, content, control) {
      systemPrompts.push(config.systemPrompt ?? '');
      control.onEvent?.({ type: 'llm.sent' });
      try {
        const stream = await provider.generate(
          config.systemPrompt ?? '',
          [...(config.tools ?? [])],
          content.messages,
          { signal: control.signal },
        );
        for await (const part of stream) {
          control.onEvent?.({ type: 'llm.streaming.part', part });
          control.signal.throwIfAborted();
        }
        control.onEvent?.({ type: 'llm.streaming.usage', usage: stream.usage ?? ZERO_USAGE });
        control.onEvent?.({
          type: 'llm.streaming.finish',
          finish: {
            finishReason: stream.finishReason,
            rawFinishReason: stream.rawFinishReason,
          },
        });
        if (stream.id !== null) {
          control.onEvent?.({ type: 'llm.streaming.message_id', messageId: stream.id });
        }
        control.onEvent?.({ type: 'llm.done' });
      } catch (error) {
        control.onEvent?.({
          type: 'llm.failed.remote',
          error: {
            kind: 'unknown',
            message: error instanceof Error ? error.message : String(error),
            retryable: false,
          },
        });
      }
    },
  };
  const real = new ProtocolAdapterRegistry();
  const registry: IProtocolAdapterRegistryType = {
    _serviceBrand: undefined,
    supportedProtocols: () => real.supportedProtocols(),
    resolveAdapterIdentity: real.resolveAdapterIdentity.bind(real),
    resolveProviderBaseId: real.resolveProviderBaseId.bind(real),
    resolveCapability: real.resolveCapability.bind(real),
    request: (model: Model) => withMockProvider(real.request(model), provider),
  };

  return {
    seed: [IProtocolAdapterRegistry, registry],
    mockNextText: (text) => {
      queue.push({ parts: [{ type: 'text', text }] });
    },
    mockNextTextWithUsage: (text, usage) => {
      queue.push({ parts: [{ type: 'text', text }], usage });
    },
    mockNextGatedText: (text, gate) => {
      queue.push({ parts: [{ type: 'text', text }], gate });
    },
    mockNextResponse: (...parts) => {
      queue.push({ parts: structuredClone(parts) });
    },
    mockNextProviderResponse: (response) => {
      queue.push({ ...response, parts: structuredClone(response.parts ?? []) });
    },
    callCount: () => calls.length,
    callHistory: () => calls,
    systemPrompts: () => systemPrompts,
  };
}
