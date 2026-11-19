/* oxlint-disable typescript-eslint/no-unsafe-declaration-merging -- Event2 class+payload-interface declaration merging is the sanctioned event-declaration idiom. */
import { z } from 'zod';

import { AgentEvent2, type AgentDomainTrait } from '#/app/event/event2';
import { defineState } from '#/state/state';
import {
  ContextApplyCompaction,
  ContextClear,
} from '#/agent/contextMemory/contextEvents';
import { UsageRecord } from '#/agent/usage/usageOps';

export type MicroCompactionReason = 'threshold' | 'idle';

export interface MicroCompactionClearedEntry {
  readonly toolCallId: string;
  readonly path: string;
  readonly tokens: number;
}

const microCompactionApplySchema = z.object({
  agentId: z.string(),
  reason: z.enum(['threshold', 'idle']),
  clearedCount: z.number().int().min(1),
  freedTokens: z.number().int().min(0),
  boundary: z.number().int().min(0),
  cleared: z.array(
    z.object({
      toolCallId: z.string(),
      path: z.string(),
      tokens: z.number().int().min(0),
    }),
  ),
});

export class MicroCompactionApply extends AgentEvent2<z.infer<typeof microCompactionApplySchema>> {
  static override readonly type = 'micro_compaction.apply';
  static override readonly durable = true;
  static override readonly schema = microCompactionApplySchema;
}
export interface MicroCompactionApply extends AgentDomainTrait {
  readonly reason: MicroCompactionReason;
  readonly clearedCount: number;
  readonly freedTokens: number;
  readonly boundary: number;
  readonly cleared: readonly MicroCompactionClearedEntry[];
}

export interface MicroCompactionState {
  readonly cleared: readonly MicroCompactionClearedEntry[];
  /**
   * Time of the last completed turn-scoped model response (from usage.record),
   * restored by session replay. Drives the idle (cache-cold) trigger, so it
   * survives restarts; reset by full compaction because the cached prefix is
   * replaced.
   */
  readonly lastTurnResponseAt: number | null;
}

const emptyMicroCompactionState = (): MicroCompactionState => ({
  cleared: [],
  lastTurnResponseAt: null,
});

export const microCompactionKey = defineState('microCompaction', emptyMicroCompactionState)
  .replayable({
    schema: z.object({
      cleared: z.array(
        z.object({
          toolCallId: z.string(),
          path: z.string(),
          tokens: z.number().int().min(0),
        }),
      ),
      // Absent in state persisted before this field existed.
      lastTurnResponseAt: z.number().nullable().default(null),
    }),
  })
  .on(MicroCompactionApply, (state, event) => {
    const known = new Set(state.cleared.map((entry) => entry.toolCallId));
    const added = event.cleared.filter((entry) => !known.has(entry.toolCallId));
    if (added.length === 0) return undefined;
    return { ...state, cleared: [...state.cleared, ...added] };
  })
  .on(UsageRecord, (state, event) => {
    if (event.usageScope !== 'turn') return undefined;
    return { ...state, lastTurnResponseAt: event.time };
  })
  .on(ContextApplyCompaction, () => emptyMicroCompactionState())
  .on(ContextClear, () => emptyMicroCompactionState());
