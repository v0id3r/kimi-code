import type { ContentPart } from '#human/llm/message';
import type { ContextMessage } from '#/agent/contextMemory/types';
import type { Message } from '#human/llm/message';
import type { MicroCompactionReason } from './microCompactionOps';
import type { MicroCompactionSettings } from './configSection';

// Prompt-cache TTL of the built-in Kimi provider (see
// human/llm/builtin/provider/kimi/protocol.ts: prompt_cache_options ttl '1h').
export const PROMPT_CACHE_TTL_MS = 60 * 60 * 1000;

// Minimum freed tokens when the trigger is the idle (cache-cold) path.
export const IDLE_MIN_FREED_TOKENS = 5000;

const READABLE_TOOL_NAMES: ReadonlySet<string> = new Set([
  'Bash',
  'Grep',
  'Glob',
  'Read',
  'FetchURL',
  'WebSearch',
  'TaskOutput',
]);

const MCP_TOOL_NAME_PREFIX = 'mcp__';

export function isMicroCompactionReadableTool(toolName: string): boolean {
  return READABLE_TOOL_NAMES.has(toolName) || toolName.startsWith(MCP_TOOL_NAME_PREFIX);
}

export function microCompactionMarker(outputPath: string): string {
  return (
    `[Old tool result content cleared. Full output saved to: ${outputPath}. ` +
    'Use Read to view it if needed.]'
  );
}

export interface MicroCompactionCandidate {
  readonly index: number;
  readonly toolCallId: string;
  readonly toolName: string;
  readonly tokens: number;
}

export function toolNamesByCallId(
  history: readonly ContextMessage[],
): ReadonlyMap<string, string> {
  const names = new Map<string, string>();
  for (const message of history) {
    if (message.role !== 'assistant') continue;
    for (const call of message.toolCalls) {
      names.set(call.id, call.name);
    }
  }
  return names;
}

/**
 * Eligible tool results: reading-only tools, large enough, not in the
 * keepRecent tail of tool results, and not already cleared.
 */
export function selectMicroCompactionCandidates(
  history: readonly ContextMessage[],
  isCleared: (toolCallId: string) => boolean,
  settings: Pick<MicroCompactionSettings, 'keepRecent' | 'minResultTokens'>,
  estimateTokens: (message: ContextMessage) => number,
): readonly MicroCompactionCandidate[] {
  const toolNames = toolNamesByCallId(history);
  const resultIndexes: number[] = [];
  for (const [index, message] of history.entries()) {
    if (message.partial === true) continue;
    if (message.role === 'tool' && message.toolCallId !== undefined) {
      resultIndexes.push(index);
    }
  }
  const protectedTail = new Set(resultIndexes.slice(-settings.keepRecent));

  const candidates: MicroCompactionCandidate[] = [];
  for (const index of resultIndexes) {
    if (protectedTail.has(index)) continue;
    const message = history[index]!;
    if (message.role !== 'tool' || message.toolCallId === undefined) continue;
    const toolCallId = message.toolCallId;
    if (isCleared(toolCallId)) continue;
    const toolName = toolNames.get(toolCallId);
    if (toolName === undefined || !isMicroCompactionReadableTool(toolName)) continue;
    const tokens = estimateTokens(message);
    if (tokens < settings.minResultTokens) continue;
    candidates.push({ index, toolCallId, toolName, tokens });
  }
  return candidates;
}

export type MicroCompactionPlan =
  | { readonly kind: 'apply'; readonly reason: MicroCompactionReason; readonly toClear: readonly MicroCompactionCandidate[]; readonly freedTokens: number }
  | { readonly kind: 'skip' };

export interface MicroCompactionPlanInput {
  readonly contextTokens: number;
  readonly cacheIdleMs: number | null;
  readonly settings: MicroCompactionSettings;
  readonly candidates: readonly MicroCompactionCandidate[];
  readonly markerTokens: number;
}

export function planMicroCompaction(input: MicroCompactionPlanInput): MicroCompactionPlan {
  const { settings, candidates, markerTokens } = input;
  if (candidates.length === 0) return { kind: 'skip' };

  const idle =
    input.cacheIdleMs !== null &&
    input.cacheIdleMs >= PROMPT_CACHE_TTL_MS + settings.idleMarginSeconds * 1000;
  if (!idle && input.contextTokens < settings.triggerTokens) return { kind: 'skip' };

  const minFreedTokens = idle ? IDLE_MIN_FREED_TOKENS : settings.clearAtLeastTokens;
  const toClear = [...candidates].toSorted((a, b) => a.index - b.index);
  const freedTokens = toClear.reduce(
    (sum, candidate) => sum + Math.max(0, candidate.tokens - markerTokens),
    0,
  );
  if (freedTokens < minFreedTokens) return { kind: 'skip' };
  return { kind: 'apply', reason: idle ? 'idle' : 'threshold', toClear, freedTokens };
}

export function toolResultText(content: readonly ContentPart[]): string {
  let text = '';
  for (const part of content) {
    if (part.type === 'text') text += part.text;
    else if (part.type === 'think') text += part.think;
  }
  return text;
}

/**
 * Project the history for an LLM request: cleared tool results are replaced
 * by the marker. Messages before the first cleared result are returned
 * untouched (same references), so the prompt-cache prefix is stable.
 */
export function projectMicroCompaction(
  messages: readonly Message[],
  clearedPaths: ReadonlyMap<string, string>,
): readonly Message[] {
  if (clearedPaths.size === 0) return messages;
  let projected: Message[] | undefined;
  for (const [index, message] of messages.entries()) {
    if (projected === undefined) {
      if (message.role !== 'tool') continue;
      const path = clearedPaths.get(message.toolCallId);
      if (path === undefined) continue;
      projected = messages.slice(0, index);
    }
    if (message.role !== 'tool') {
      projected.push(message);
      continue;
    }
    const path = clearedPaths.get(message.toolCallId);
    projected.push(
      path === undefined
        ? message
        : { ...message, content: [{ type: 'text', text: microCompactionMarker(path) }] },
    );
  }
  return projected ?? messages;
}
