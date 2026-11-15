import { describe, expect, it } from 'vitest';

import {
  kimiAnthropicProtocol,
  kimiOpenAIProtocol,
  kimiResponsesProtocol,
} from '#human/llm/builtin/provider/kimi/protocol';

const model = { spec: { provider: 'kimi', name: 'kimi-k3.1-pro' } };
const content = {
  messages: [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'hello' }] }],
};

describe('Kimi prompt cache TTL', () => {
  for (const [name, protocol, field, expected] of [
    [
      'chat completions',
      kimiOpenAIProtocol,
      'prompt_cache_options',
      { mode: 'implicit', ttl: '1h' },
    ],
    ['responses', kimiResponsesProtocol, 'prompt_cache_options', { mode: 'implicit', ttl: '1h' }],
    [
      'anthropic messages',
      kimiAnthropicProtocol,
      'cache_control',
      { type: 'ephemeral', ttl: '1h' },
    ],
  ] as const) {
    it(`sets one-hour caching for ${name}`, () => {
      const request = protocol.encode({ model, profile: {}, content, options: undefined });

      expect(request.params).toMatchObject({ [field]: expected });
    });
  }

  it('allows chat completions extras to select five-minute caching', () => {
    const request = kimiOpenAIProtocol.encode({
      model,
      profile: {},
      content,
      options: {
        extras: {
          extra_body: { prompt_cache_options: { mode: 'implicit', ttl: '5m' } },
        },
      },
    });

    expect(request.params).toMatchObject({
      prompt_cache_options: { mode: 'implicit', ttl: '5m' },
    });
  });
});
