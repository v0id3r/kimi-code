/* oxlint-disable typescript-eslint/no-unsafe-declaration-merging -- Event2 class+payload-interface declaration merging is the sanctioned event-declaration idiom. */
import { z } from 'zod';

import { AgentEvent2, type AgentDomainTrait } from '#/app/event/event2';
import { defineState } from '#/state/state';
import {
  ContextApplyCompaction,
  ContextClear,
} from '#/agent/contextMemory/contextEvents';

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
}

const emptyMicroCompactionState = (): MicroCompactionState => ({ cleared: [] });

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
    }),
  })
  .on(MicroCompactionApply, (state, event) => {
    const known = new Set(state.cleared.map((entry) => entry.toolCallId));
    const added = event.cleared.filter((entry) => !known.has(entry.toolCallId));
    if (added.length === 0) return undefined;
    return { cleared: [...state.cleared, ...added] };
  })
  .on(ContextApplyCompaction, () => emptyMicroCompactionState())
  .on(ContextClear, () => emptyMicroCompactionState());
