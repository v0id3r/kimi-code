import { describe, expect, it } from 'vitest';

import type { ContextMessage } from '#/agent/contextMemory/types';
import type { Message } from '#human/llm/message';
import {
  IDLE_MIN_FREED_TOKENS,
  isMicroCompactionReadableTool,
  microCompactionMarker,
  planMicroCompaction,
  projectMicroCompaction,
  selectMicroCompactionCandidates,
  toolResultText,
  PROMPT_CACHE_TTL_MS,
} from '#/agent/microCompaction/microCompactionCore';
import { resolveMicroCompactionSettings } from '#/agent/microCompaction/configSection';

const SETTINGS = resolveMicroCompactionSettings({
  keepRecent: 2,
  minResultTokens: 100,
  triggerTokens: 1000,
  clearAtLeastTokens: 500,
  idleMarginSeconds: 300,
});

const estimateTokens = (message: ContextMessage): number =>
  message.role === 'tool' ? toolResultText(message.content).length : 0;

function assistant(callId: string, toolName: string): ContextMessage {
  return {
    role: 'assistant',
    content: [],
    toolCalls: [{ type: 'function', id: callId, name: toolName, arguments: null }],
  };
}

function toolResult(callId: string, text: string): ContextMessage {
  return { role: 'tool', content: [{ type: 'text', text }], toolCallId: callId };
}

function exchange(toolName: string, callId: string, text: string): ContextMessage[] {
  return [assistant(callId, toolName), toolResult(callId, text)];
}

describe('isMicroCompactionReadableTool', () => {
  it('clears reading tools and MCP tools only', () => {
    for (const name of ['Bash', 'Grep', 'Glob', 'Read', 'FetchURL', 'WebSearch', 'TaskOutput']) {
      expect(isMicroCompactionReadableTool(name)).toBe(true);
    }
    expect(isMicroCompactionReadableTool('mcp__github__create_issue')).toBe(true);
    for (const name of ['Edit', 'Write', 'TodoList', 'AskUserQuestion', 'Skill', 'Agent', 'Task']) {
      expect(isMicroCompactionReadableTool(name)).toBe(false);
    }
  });
});

describe('selectMicroCompactionCandidates', () => {
  it('selects large old results of reading tools', () => {
    const history: ContextMessage[] = [
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      ...exchange('Bash', 'c1', 'x'.repeat(150)),
      ...exchange('Grep', 'c2', 'y'.repeat(10)),
      ...exchange('Bash', 'c4', 'w'.repeat(150)),
      ...exchange('Read', 'c3', 'z'.repeat(30)),
    ];
    // keepRecent = 2 protects c4 and c3 (the two newest results); c2 is
    // below minResultTokens.
    const candidates = selectMicroCompactionCandidates(history, () => false, SETTINGS, estimateTokens);
    expect(candidates.map((c) => c.toolCallId)).toEqual(['c1']);
    expect(candidates[0]).toMatchObject({ index: 2, toolName: 'Bash' });
  });

  it('never selects non-reading tools or unknown call ids', () => {
    const history: ContextMessage[] = [
      ...exchange('Edit', 'c1', 'x'.repeat(150)),
      ...exchange('Write', 'c2', 'y'.repeat(150)),
      ...exchange('Agent', 'c3', 'z'.repeat(150)),
      { role: 'tool', content: [{ type: 'text', text: 'x'.repeat(150) }], toolCallId: 'orphan' },
    ];
    const candidates = selectMicroCompactionCandidates(
      history,
      () => false,
      { ...SETTINGS, keepRecent: 0 },
      estimateTokens,
    );
    expect(candidates).toEqual([]);
  });

  it('protects the keepRecent tail of tool results', () => {
    const history: ContextMessage[] = [
      ...exchange('Bash', 'c1', 'x'.repeat(150)),
      ...exchange('Bash', 'c2', 'x'.repeat(150)),
      ...exchange('Bash', 'c3', 'x'.repeat(150)),
      ...exchange('Bash', 'c4', 'x'.repeat(150)),
    ];
    const candidates = selectMicroCompactionCandidates(history, () => false, SETTINGS, estimateTokens);
    expect(candidates.map((c) => c.toolCallId)).toEqual(['c1', 'c2']);
  });

  it('applies the minResultTokens floor', () => {
    const history: ContextMessage[] = [
      ...exchange('Bash', 'c1', 'x'.repeat(150)),
      ...exchange('Bash', 'c2', 'x'.repeat(150)),
      ...exchange('Bash', 'c3', 'x'),
    ];
    const candidates = selectMicroCompactionCandidates(history, () => false, SETTINGS, estimateTokens);
    expect(candidates.map((c) => c.toolCallId)).toEqual(['c1']);
  });

  it('monotonicity: excludes already cleared results', () => {
    const history: ContextMessage[] = [
      ...exchange('Bash', 'c1', 'x'.repeat(150)),
      ...exchange('Bash', 'c2', 'x'.repeat(150)),
      ...exchange('Bash', 'c3', 'x'.repeat(150)),
      ...exchange('Bash', 'c4', 'x'.repeat(150)),
    ];
    const first = selectMicroCompactionCandidates(history, () => false, SETTINGS, estimateTokens);
    const cleared = new Set(first.map((c) => c.toolCallId));
    const second = selectMicroCompactionCandidates(
      history,
      (id) => cleared.has(id),
      SETTINGS,
      estimateTokens,
    );
    expect(second).toEqual([]);
    for (const candidate of first) {
      expect(cleared.has(candidate.toolCallId)).toBe(true);
    }
  });
});

describe('planMicroCompaction', () => {
  const candidates = [
    { index: 1, toolCallId: 'c1', toolName: 'Bash', tokens: 400 },
    { index: 3, toolCallId: 'c2', toolName: 'Read', tokens: 400 },
  ];

  it('skips when context is below trigger and cache is warm', () => {
    const plan = planMicroCompaction({
      contextTokens: 500,
      cacheIdleMs: 1000,
      settings: SETTINGS,
      candidates,
      markerTokens: 10,
    });
    expect(plan).toEqual({ kind: 'skip' });
  });

  it('applies with threshold reason when context exceeds trigger', () => {
    const plan = planMicroCompaction({
      contextTokens: 2000,
      cacheIdleMs: 1000,
      settings: SETTINGS,
      candidates,
      markerTokens: 10,
    });
    expect(plan).toMatchObject({ kind: 'apply', reason: 'threshold', freedTokens: 780 });
  });

  it('applies with idle reason below the context trigger when the cache is cold', () => {
    const plan = planMicroCompaction({
      contextTokens: 500,
      cacheIdleMs: PROMPT_CACHE_TTL_MS + 300_000 + 1,
      settings: SETTINGS,
      candidates: [
        { index: 1, toolCallId: 'c1', toolName: 'Bash', tokens: 3000 },
        { index: 3, toolCallId: 'c2', toolName: 'Read', tokens: 3000 },
      ],
      markerTokens: 10,
    });
    expect(plan).toMatchObject({ kind: 'apply', reason: 'idle' });
  });

  it('requires the minimum freed tokens', () => {
    const plan = planMicroCompaction({
      contextTokens: 2000,
      cacheIdleMs: 1000,
      settings: SETTINGS,
      candidates: [{ index: 1, toolCallId: 'c1', toolName: 'Bash', tokens: SETTINGS.clearAtLeastTokens - 1 }],
      markerTokens: 10,
    });
    expect(plan).toEqual({ kind: 'skip' });
  });

  it('idle path uses the smaller idle floor', () => {
    const small = [{ index: 1, toolCallId: 'c1', toolName: 'Bash', tokens: IDLE_MIN_FREED_TOKENS + 100 }];
    const plan = planMicroCompaction({
      contextTokens: 500,
      cacheIdleMs: PROMPT_CACHE_TTL_MS + 300_000 + 1,
      settings: SETTINGS,
      candidates: small,
      markerTokens: 10,
    });
    expect(plan).toMatchObject({ kind: 'apply', reason: 'idle' });
  });

  it('idle path still respects its minimum freed tokens', () => {
    const plan = planMicroCompaction({
      contextTokens: 500,
      cacheIdleMs: PROMPT_CACHE_TTL_MS + 300_000 + 1,
      settings: SETTINGS,
      candidates: [{ index: 1, toolCallId: 'c1', toolName: 'Bash', tokens: IDLE_MIN_FREED_TOKENS - 1 }],
      markerTokens: 10,
    });
    expect(plan).toEqual({ kind: 'skip' });
  });

  it('skips without candidates', () => {
    expect(
      planMicroCompaction({
        contextTokens: 999999,
        cacheIdleMs: null,
        settings: SETTINGS,
        candidates: [],
        markerTokens: 10,
      }),
    ).toEqual({ kind: 'skip' });
  });
});

describe('projectMicroCompaction', () => {
  const messages: Message[] = [
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'thinking' }], toolCalls: [] },
    { role: 'tool', content: [{ type: 'text', text: 'old bash output' }], toolCallId: 'c1' },
    { role: 'tool', content: [{ type: 'text', text: 'keep me' }], toolCallId: 'c2' },
    { role: 'assistant', content: [{ type: 'text', text: 'later' }], toolCalls: [] },
  ];
  const clearedPaths = new Map([['c1', '/tmp/results/c1.txt']]);

  it('replaces cleared results with the marker and keeps everything else', () => {
    const projected = projectMicroCompaction(messages, clearedPaths);
    expect(projected).toHaveLength(messages.length);
    const cleared = projected[2]!;
    expect(cleared.role).toBe('tool');
    if (cleared.role === 'tool') {
      expect(cleared.content).toEqual([
        { type: 'text', text: microCompactionMarker('/tmp/results/c1.txt') },
      ]);
    }
    expect(projected[3]).toBe(messages[3]);
    expect(projected[4]).toBe(messages[4]);
  });

  it('keeps references identical up to the first cleared result', () => {
    const projected = projectMicroCompaction(messages, clearedPaths);
    expect(projected[0]).toBe(messages[0]);
    expect(projected[1]).toBe(messages[1]);
    expect(projected[2]).not.toBe(messages[2]);
  });

  it('returns the input unchanged when nothing is cleared', () => {
    expect(projectMicroCompaction(messages, new Map())).toBe(messages);
  });

  it('marker mentions the saved path and Read', () => {
    expect(microCompactionMarker('/a/b.txt')).toBe(
      '[Old tool result content cleared. Full output saved to: /a/b.txt. Use Read to view it if needed.]',
    );
  });
});

describe('toolResultText', () => {
  it('concatenates text and think parts', () => {
    expect(
      toolResultText([
        { type: 'text', text: 'a' },
        { type: 'think', think: 'b' },
      ]),
    ).toBe('ab');
  });
});
