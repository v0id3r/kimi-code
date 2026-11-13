import {
  IProtocolAdapterRegistry,
  type IProtocolAdapterRegistry as IProtocolAdapterRegistryType,
  type Model,
  ProtocolAdapterRegistry,
} from '@moonshot-ai/agent-core-v2';
import type { Message, StreamedMessagePart } from '@moonshot-ai/core/llm/message';
import type { FinishReason } from '@moonshot-ai/core/llm/response';
import {
  createMockProvider,
  withMockProvider,
  type MockProviderResponse,
} from '../../../agent-core-v2/test/harness/scripted-generate.js';

export interface ScriptedProvider {
  readonly seed: readonly [typeof IProtocolAdapterRegistry, IProtocolAdapterRegistryType];
  mockNextText(text: string): void;
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
