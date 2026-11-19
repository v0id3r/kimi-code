import { randomUUID } from 'node:crypto';

import { IInstantiationService } from '#/_base/di/instantiation';
import { Service } from '#/_base/di/service';
import { LifecycleScope } from '#/app/scopes';
import { ScopeActivation, registerScopedService } from '#/_base/di/scope';
import { IConfigService } from '#/app/config/config';
import { IBootstrapService } from '#/app/bootstrap/bootstrap';
import { ILogService } from '#/_base/log/log';
import { IFileSystemStorageService } from '#/persistence/interface/storage';
import { IEventDispatcher } from '#/state/eventDispatcher';
import { ISessionTokenCountingService } from '#/session/tokenCounting/sessionTokenCounting';
import { agentContextOfScope, IAgentScopeContext } from '#/agent/scopeContext/scopeContext';
import { IAgentStateService } from '#/agent/state/agentState';
import { IAgentLoopService } from '#/agent/loop/loop';
import type { Message } from '#human/llm/message';
import { join } from 'pathe';

import { IAgentMicroCompactionService } from './microCompaction';
import {
  MICRO_COMPACTION_SECTION,
  resolveMicroCompactionSettings,
  type MicroCompactionConfig,
  type MicroCompactionSettings,
} from './configSection';
import {
  microCompactionKey,
  MicroCompactionApply,
  type MicroCompactionClearedEntry,
} from './microCompactionOps';
import {
  microCompactionMarker,
  planMicroCompaction,
  projectMicroCompaction,
  selectMicroCompactionCandidates,
  toolResultText,
} from './microCompactionCore';

const encoder = new TextEncoder();

export class AgentMicroCompactionService extends Service implements IAgentMicroCompactionService {
  declare readonly _serviceBrand: undefined;

  private readonly storageScope: string;

  constructor(
    @IConfigService private readonly configService: IConfigService,
    @IInstantiationService private readonly instantiation: IInstantiationService,
    @ISessionTokenCountingService private readonly tokenCounting: ISessionTokenCountingService,
    @IAgentScopeContext private readonly agent: IAgentScopeContext,
    @IAgentStateService private readonly states: IAgentStateService,
    @IEventDispatcher private readonly dispatcher: IEventDispatcher,
    @IFileSystemStorageService private readonly storage: IFileSystemStorageService,
    @IBootstrapService private readonly bootstrap: IBootstrapService,
    @ILogService private readonly log: ILogService,
  ) {
    super();
    this.storageScope = agent.scope('tool-results');
    this.states.contributeState(microCompactionKey);
  }

  private settings(): MicroCompactionSettings | undefined {
    const resolved = resolveMicroCompactionSettings(
      this.configService.get<MicroCompactionConfig>(MICRO_COMPACTION_SECTION),
    );
    return resolved.enabled ? resolved : undefined;
  }

  private clearedEntries(): readonly MicroCompactionClearedEntry[] {
    return this.states.get(microCompactionKey).cleared;
  }

  active(): boolean {
    return this.settings() !== undefined && this.clearedEntries().length > 0;
  }

  projectMessages(messages: readonly Message[]): readonly Message[] {
    const cleared = this.clearedEntries();
    if (cleared.length === 0) return messages;
    const clearedPaths = new Map(cleared.map((entry) => [entry.toolCallId, entry.path]));
    return projectMicroCompaction(messages, clearedPaths);
  }

  async prepareForTurnRequest(now: number = Date.now()): Promise<void> {
    const settings = this.settings();
    if (settings === undefined) return;

    const history = this.instantiation.invokeFunction((accessor) =>
      accessor.get(IAgentLoopService).history(),
    );
    const clearedIds = new Set(this.clearedEntries().map((entry) => entry.toolCallId));
    const candidates = selectMicroCompactionCandidates(
      history,
      (toolCallId) => clearedIds.has(toolCallId),
      settings,
      (message) => this.tokenCounting.estimateMessage(message),
    );
    const lastTurnResponseAt = this.states.get(microCompactionKey).lastTurnResponseAt;
    const cacheIdleMs = lastTurnResponseAt === null ? null : now - lastTurnResponseAt;
    const plan = planMicroCompaction({
      contextTokens: this.tokenCounting.get(agentContextOfScope(this.agent)).size,
      cacheIdleMs,
      settings,
      candidates,
      markerTokens: this.tokenCounting.estimateText(microCompactionMarker('')),
    });
    if (plan.kind === 'skip') return;

    const entries: MicroCompactionClearedEntry[] = [];
    for (const candidate of plan.toClear) {
      const message = history[candidate.index];
      if (message?.role !== 'tool') continue;
      const path = await this.saveToolResult(candidate.toolName, candidate.toolCallId, toolResultText(message.content));
      if (path !== undefined) {
        entries.push({ toolCallId: candidate.toolCallId, path, tokens: candidate.tokens });
      }
    }
    if (entries.length === 0) return;

    const boundary = Math.max(...entries.map((entry) => {
      const candidate = plan.toClear.find((item) => item.toolCallId === entry.toolCallId);
      return candidate?.index ?? 0;
    }));
    await this.dispatcher.dispatch(
      new MicroCompactionApply({
        agentId: this.agent.agentId,
        reason: plan.reason,
        clearedCount: entries.length,
        freedTokens: plan.freedTokens,
        boundary,
        cleared: entries,
      }),
    );
  }

  private async saveToolResult(
    toolName: string,
    toolCallId: string,
    text: string,
  ): Promise<string | undefined> {
    const key = `${safeToolResultFileStem(toolName, toolCallId)}-${randomUUID()}.txt`;
    const path = join(this.bootstrap.homeDir, this.storageScope, key);
    try {
      await this.storage.write(this.storageScope, key, encoder.encode(text), { atomic: true });
      return path;
    } catch (error) {
      this.log.warn('micro_compaction: failed to save cleared tool result; keeping it in context', {
        agentId: this.agent.agentId,
        toolCallId,
        path,
        error: error instanceof Error ? error.message : String(error),
      });
      return undefined;
    }
  }
}

function safeToolResultFileStem(toolName: string, toolCallId: string): string {
  const label = `${toolName}-${toolCallId}`
    .replaceAll(/[^a-zA-Z0-9._-]+/g, '_')
    .replaceAll(/^_+|_+$/g, '')
    .slice(0, 80);
  return label || 'tool-result';
}

registerScopedService(
  LifecycleScope.Agent,
  IAgentMicroCompactionService,
  AgentMicroCompactionService,
  ScopeActivation.OnScopeCreated,
  'microCompaction',
);
