import { createDecorator } from '#/_base/di/instantiation';
import type { Message } from '#human/llm/message';

export const IAgentMicroCompactionService = createDecorator<IAgentMicroCompactionService>(
  'agentMicroCompactionService',
);

export interface IAgentMicroCompactionService {
  readonly _serviceBrand: undefined;
  /**
   * Micro-compaction is active: config enabled and at least one result has
   * been cleared (so projection has work to do).
   */
  active(): boolean;
  /**
   * Evaluate triggers against the current history, persist cleared outputs to
   * files, and record the (monotonically growing) cleared set. Called when a
   * turn request is assembled.
   */
  prepareForTurnRequest(now?: number): Promise<void>;
  /** Project request messages, replacing cleared tool results with markers. */
  projectMessages(messages: readonly Message[]): readonly Message[];
}
