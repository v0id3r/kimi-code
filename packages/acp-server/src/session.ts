import { randomUUID } from 'node:crypto';

import type {
  AvailableCommand,
  ContentBlock,
  PromptResponse,
  SessionConfigOption,
  SessionModeState,
  SessionNotification,
  ToolCallContent,
  ToolCallLocation,
} from '@agentclientprotocol/sdk';
import { RequestError } from '@agentclientprotocol/sdk';
import {
  grandTotal,
  type ContextMessage,
  type TokenUsage,
} from '@moonshot-ai/agent-core-v2';
import { toCoreMessage } from '@moonshot-ai/agent-core-v2/human/message';
import { toLlmMessage } from '@moonshot-ai/agent-core-v2/llm-adapter/contract/message';
import type {
  AgentEventPayloads,
  AgentHandle,
  ContentPart,
  IDisposable,
  Klient,
  PromptLaunchResult,
  SessionEventPayloads,
  SessionHandle,
  SkillSummary,
  SteerResult,
} from '@moonshot-ai/klient';
import type { ToolResultEvent } from '@moonshot-ai/agent-core-v2/events';
import type {
  ToolCallDeltaEvent,
  ToolCallStartedEvent,
  ToolUpdatedEvent,
} from '@moonshot-ai/agent-core-v2/agent/toolExecutor/toolExecutorEvents';
import type { ToolInputDisplay } from '@moonshot-ai/agent-core-v2/tool/toolInputDisplay';

import type { AcpClient } from './acp-client';
import type { AcpTerminalCreatedEvent, IAcpConnection } from './acp-fs';
import {
  ACP_BUILTIN_SLASH_COMMAND_NAMES,
  ACP_BUILTIN_SLASH_COMMANDS,
  type AcpBuiltinSlashCommandName,
  runBuiltinSlashCommand,
} from './builtin-commands';
import { buildSessionConfigOptions } from './config-options';
import {
  acpBlocksToContentParts,
  compressPromptImageParts,
  displayBlockToAcpContent,
} from './convert';
import {
  acpToolCallId,
  assistantDeltaToSessionUpdate,
  asyncTaskState,
  availableCommandsUpdateNotification,
  configOptionUpdateNotification,
  currentModeUpdateNotification,
  planFromDisplayBlock,
  sessionInfoUpdateNotification,
  thinkingDeltaToSessionUpdate,
  toolCallDeltaToSessionUpdate,
  toolCallLazyCreateToSessionUpdate,
  toolCallLocations,
  toolCallStartedUpgradeToSessionUpdate,
  toolCallStartToSessionUpdate,
  toolUpdatedToSessionUpdate,
  toolResultToSessionUpdate,
  turnEndReasonToStopReason,
  usageUpdateNotification,
  type AcpUsageReport,
  isAuthError,
  stringifyArgs,
} from './events-map';
import { AcpInteractionBridge } from './interaction-bridge';
import { log } from './log';
import { projectModelCatalog } from './model-catalog';
import { ACP_MODES, type AcpModeId, acpModeToToggles, DEFAULT_MODE_ID } from './modes';
import { projectHistoryToSessionUpdates } from './replay';
import { buildAcpSkillSlashCommands, detectSlashIntent } from './slash';

function leadingText(blocks: readonly ContentBlock[]): string | undefined {
  const first = blocks[0];
  if (first !== undefined && first.type === 'text') return first.text;
  return undefined;
}

function subtractUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    inputOther: a.inputOther - b.inputOther,
    output: a.output - b.output,
    inputCacheRead:
      a.inputCacheRead === undefined || b.inputCacheRead === undefined
        ? undefined
        : a.inputCacheRead - b.inputCacheRead,
    inputCacheCreation: a.inputCacheCreation - b.inputCacheCreation,
  };
}

const TURN_AGENT_BUSY_CODE = 'turn.agent_busy';
const SESSION_OCCUPIED_CODE = 'session.occupied';
const SESSION_STALE_CODE = 'session.stale';

export function mapPromptLaunchError(error: unknown, sessionId: string): RequestError {
  const code = (error as { readonly code?: unknown } | null | undefined)?.code;
  const message = error instanceof Error ? error.message : String(error);
  if (typeof code === 'string' && isAuthError({ code })) {
    log.warn('acp: prompt launch rejected with an auth error; mapping to auth_required', {
      sessionId,
      error: message,
    });
    return RequestError.authRequired(undefined, message);
  }
  if (code === TURN_AGENT_BUSY_CODE) {
    log.warn('acp: prompt rejected because another turn is active', { sessionId });
    return RequestError.invalidRequest({ code }, message);
  }
  if (code === SESSION_OCCUPIED_CODE) {
    log.warn('acp: prompt rejected because another client holds the session', { sessionId });
    return RequestError.invalidRequest({ code }, message);
  }
  if (code === SESSION_STALE_CODE) {
    log.warn('acp: prompt rejected because the session was updated elsewhere', { sessionId });
    return RequestError.invalidRequest(
      { code },
      'Session was updated by another client. Close and reopen the session, then retry.',
    );
  }
  log.error('acp: prompt launch failed', {
    sessionId,
    error: error instanceof Error ? { message: error.message, stack: error.stack } : String(error),
  });
  return RequestError.internalError(undefined, 'session prompt failed');
}

interface HostSlashCommandsSnapshot {
  readonly commands: ReadonlyArray<AvailableCommand>;
  readonly skillCommandMap?: ReadonlyMap<string, string>;
}

interface ResolvedCommands {
  readonly commands: AvailableCommand[];
  readonly skillCommandMap: ReadonlyMap<string, string>;
}

export interface SteeringResponse {
  readonly outcome: 'injected' | 'startedNewTurn';
  readonly stopReason: string;
}

interface SteerWaiter {
  promptId?: string;
  turnId?: number;
  awaitingNextTurn: boolean;
  settled: boolean;
  settle: (stopReason: string) => void;
}

interface TurnDriver {
  resolve(response: PromptResponse): void;
  reject(error: unknown): void;
  promptId?: string;
  turnId?: number;
  settled: boolean;
  cancelRequested?: boolean;
  launchPending: boolean;
  queued: boolean;
  awaitingResponseTurn: boolean;
  early: Array<() => void>;
}

type TurnEnded = AgentEventPayloads['turn.ended'];
type TaskEvent = AgentEventPayloads['task.started'];

export class AcpSession {
  private readonly klient: Klient;
  private readonly session: SessionHandle;
  private readonly agent: AgentHandle;

  private currentModelId: string = '';
  private currentThinkingLevel: string = 'off';
  private reportedUsage: TokenUsage | undefined;
  private currentModeId: AcpModeId = DEFAULT_MODE_ID;
  private skills: readonly SkillSummary[] = [];
  private driver: TurnDriver | undefined;
  private readonly steerWaiters = new Set<SteerWaiter>();
  private work: { readonly owned: boolean } | undefined;
  private lastEnd: TurnEnded | undefined;
  private readonly takenIn: string[] = [];
  private readonly stoppable = new Set<string>();
  private promptChain: Promise<unknown> = Promise.resolve();
  private readonly chainedPrompts = new Set<{ cancelled: boolean }>();
  private readonly pendingPromptAborts = new Set<{ aborted: boolean }>();
  private readonly subscriptions: IDisposable[] = [];
  private readonly toolCallStreamArgs = new Map<string, { args: string }>();
  private readonly toolStatuses = new Map<string, string>();
  private readonly toolLocations = new Map<string, ToolCallLocation[]>();
  private readonly toolDiffs = new Map<string, ToolCallContent>();
  private readonly bashCallsAwaitingTerminal = new Map<string, string>();
  private readonly terminalBackedCalls = new Map<string, string>();
  private readonly interactionBridge: AcpInteractionBridge;

  constructor(
    private readonly conn: AcpClient,
    klient: Klient,
    readonly sessionId: string,
    private readonly acpConnection: IAcpConnection,
    elicitationForm: boolean,
    private readonly resolveOriginalsDir?: (sessionId: string) => string | undefined,
    private readonly hostCommands:
      | ReadonlyArray<AvailableCommand>
      | HostSlashCommandsSnapshot = [],
  ) {
    this.klient = klient;
    this.session = klient.session(sessionId);
    this.agent = this.session.agent('main');
    this.interactionBridge = new AcpInteractionBridge(conn, this.session, sessionId, elicitationForm);
  }

  async init(): Promise<void> {
    const events = this.agent.events;
    this.subscriptions.push(
      events.on('assistant.delta', (event) => {
        this.dispatchTurnEvent(() => {
          this.onAssistantDelta(event);
        });
      }),
      events.on('thinking.delta', (event) => {
        this.dispatchTurnEvent(() => {
          this.onThinkingDelta(event);
        });
      }),
      events.on('tool.call.started', (event) => {
        this.dispatchTurnEvent(() => {
          this.onToolCallStarted(event);
        });
      }),
      events.on('tool.call.delta', (event) => {
        this.dispatchTurnEvent(() => {
          this.onToolCallDelta(event);
        });
      }),
      events.on('tool.updated', (event) => {
        this.dispatchTurnEvent(() => {
          this.onToolUpdated(event);
        });
      }),
      events.on('tool.result', (event) => {
        this.dispatchTurnEvent(() => {
          this.onToolResult(event);
        });
      }),
      events.on('turn.started', (event) => {
        this.onTurnStarted(event);
      }),
      events.on('turn.ended', (event) => {
        this.dispatchTurnEvent(() => {
          this.onTurnEnded(event);
        });
      }),
      events.on('task.started', ({ info }) => {
        const canStop = info.swarmIndex === undefined;
        if (canStop) this.stoppable.add(info.taskId);
        this.reportTask({ sessionUpdate: 'async_task_spawned', name: info.description, taskType: info.kind, canStop }, info);
      }),
      events.on('task.terminated', ({ info }) => {
        this.stoppable.delete(info.taskId);
        this.reportTask({ sessionUpdate: 'async_task_state_update', state: asyncTaskState(info.status) }, info);
      }),
      events.on('prompt.completed', (event) => {
        this.onPromptCompleted(event);
      }),
      events.on('prompt.aborted', (event) => {
        this.onPromptAborted(event);
      }),
      events.on('compaction.started', (event) => {
        this.onCompactionStarted(event);
      }),
      events.on('compaction.completed', (event) => {
        this.onCompactionCompleted(event);
      }),
      events.on('compaction.cancelled', () => {
        this.emitLocalChunk('Compaction cancelled.');
      }),
      events.on('compaction.blocked', () => {
        this.emitLocalChunk(
          'Compaction is blocked by the current turn; retry when the turn is idle.',
        );
      }),
    );
    this.subscriptions.push(
      this.session.events.on('metadata.changed', (event) => {
        this.onMetadataChanged(event);
      }),
    );
    this.subscriptions.push(
      this.session.events.on('skills.changed', () => {
        void this.refreshSkills().then(() => this.emitAvailableCommandsUpdate());
      }),
    );
    const unsubscribeTerminal = this.acpConnection.onTerminalCreated((event) => {
      this.onTerminalCreated(event);
    });
    this.subscriptions.push({ dispose: unsubscribeTerminal });
    try {
      this.currentModelId = await this.agent.getModel();
      this.currentThinkingLevel = await this.agent.getThinking();
      this.reportedUsage = (await this.agent.getUsage()).total;
    } catch (error) {
      log.warn('acp: could not seed model/thinking/usage state', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    await this.refreshSkills();
  }

  private async refreshSkills(): Promise<void> {
    try {
      this.skills = await this.session.skills.list();
    } catch (error) {
      log.warn('acp: could not list session skills', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  dispose(): void {
    this.cancel();
    const driver = this.driver;
    if (driver !== undefined) {
      this.settleDriver(driver, () => {
        driver.resolve({ stopReason: 'cancelled' });
      });
    }
    for (const waiter of this.steerWaiters) {
      waiter.settle('cancelled');
    }
    this.interactionBridge.dispose();
    for (const subscription of this.subscriptions.splice(0)) {
      subscription.dispose();
    }
  }

  async replayHistory(): Promise<void> {
    let messages: readonly ContextMessage[];
    try {
      messages = (await this.agent.getContext()).history.map((message) => ({
        ...toCoreMessage(toLlmMessage(message)),
        id: message.id,
        providerMessageId: message.providerMessageId,
        name: message.name,
        origin: message.origin,
        isError: message.isError,
        partial: message.partial,
        toolCallDisplays: message.toolCallDisplays,
        note: message.note,
        usage: message.usage,
        llmTiming: message.llmTiming,
        durationMs: message.durationMs,
      }));
    } catch (error) {
      log.warn('acp: replayHistory could not read context memory', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    const updates = projectHistoryToSessionUpdates(this.sessionId, messages);
    for (const update of updates) {
      try {
        await this.conn.sessionUpdate(update);
      } catch (error) {
        log.warn('acp: replayHistory failed to push a session/update; continuing', {
          sessionId: this.sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  private resolveCommands(): ResolvedCommands {
    const skillSnapshot = buildAcpSkillSlashCommands(this.skills);
    const hostSnapshot = Array.isArray(this.hostCommands)
      ? { commands: this.hostCommands, skillCommandMap: new Map<string, string>() }
      : (this.hostCommands as HostSlashCommandsSnapshot);
    const commands: AvailableCommand[] = [];
    const names = new Set<string>();
    for (const command of [
      ...ACP_BUILTIN_SLASH_COMMANDS,
      ...skillSnapshot.commands,
      ...hostSnapshot.commands,
    ]) {
      if (names.has(command.name)) continue;
      names.add(command.name);
      commands.push(command);
    }
    const commandMap = new Map(skillSnapshot.commandMap);
    for (const [alias, skillName] of hostSnapshot.skillCommandMap ?? []) {
      if (ACP_BUILTIN_SLASH_COMMAND_NAMES.has(alias)) continue;
      if (!names.has(alias) || commandMap.has(alias)) continue;
      commandMap.set(alias, skillName);
    }
    return { commands, skillCommandMap: commandMap };
  }

  availableCommands(): AvailableCommand[] {
    return this.resolveCommands().commands;
  }

  async emitAvailableCommandsUpdate(): Promise<void> {
    try {
      const { commands } = this.resolveCommands();
      await this.conn.sessionUpdate(availableCommandsUpdateNotification(this.sessionId, commands));
    } catch (error) {
      log.warn('acp: failed to push available_commands_update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async prompt(blocks: readonly ContentBlock[]): Promise<PromptResponse> {
    const text = leadingText(blocks);
    if (text !== undefined) {
      const commands = this.resolveCommands();
      const intent = detectSlashIntent(text, commands.skillCommandMap);
      if (intent.kind === 'builtin') {
        return this.driveBuiltinCommand(intent.name, intent.args, commands.commands);
      }
      if (intent.kind === 'skill') {
        return this.driveSkillActivation(intent.skillName, intent.args);
      }
      if (intent.kind === 'unknown') {
        return this.driveUnknownCommand(intent.name);
      }
    }

    const content = await this.preparePromptContent(blocks);
    if (content === undefined) {
      return { stopReason: 'cancelled' };
    }
    return this.enqueuePrompt(() => this.driveTurn(content));
  }

  private enqueuePrompt(task: () => Promise<PromptResponse>): Promise<PromptResponse> {
    const marker = { cancelled: false };
    this.chainedPrompts.add(marker);
    const run = this.promptChain.then(() => {
      this.chainedPrompts.delete(marker);
      if (marker.cancelled) return { stopReason: 'cancelled' } as PromptResponse;
      return task();
    });
    this.promptChain = run.catch(() => undefined);
    return run;
  }

  async stopTask(taskId: string): Promise<boolean> {
    if (!this.stoppable.has(taskId)) return false;
    await this.agent.stopTask({ taskId });
    return true;
  }

  async steer(blocks: readonly ContentBlock[], waitForCompletion: boolean): Promise<SteeringResponse> {
    const content = await this.preparePromptContent(blocks);
    if (content === undefined) {
      return { outcome: 'startedNewTurn', stopReason: 'cancelled' };
    }
    let result: SteerResult;
    try {
      result = await this.agent.steer({ input: content });
    } catch (error) {
      throw mapPromptLaunchError(error, this.sessionId);
    }
    if (result.state === 'blocked') {
      throw RequestError.invalidRequest({ code: 'prompt.blocked' }, 'the prompt was blocked');
    }
    const outcome = result.steered === true ? 'injected' : 'startedNewTurn';
    if (!waitForCompletion) {
      return { outcome, stopReason: 'end_turn' };
    }
    return new Promise<SteeringResponse>((resolve) => {
      const waiter: SteerWaiter = {
        promptId: result.prompt_id,
        turnId: undefined,
        awaitingNextTurn: false,
        settled: false,
        settle: (stopReason) => {
          if (waiter.settled) return;
          waiter.settled = true;
          this.steerWaiters.delete(waiter);
          resolve({ outcome, stopReason });
        },
      };
      this.steerWaiters.add(waiter);
    });
  }

  private async preparePromptContent(
    blocks: readonly ContentBlock[],
  ): Promise<readonly ContentPart[] | undefined> {
    const pending = { aborted: false };
    this.pendingPromptAborts.add(pending);
    let content: readonly ContentPart[];
    try {
      content = await compressPromptImageParts(acpBlocksToContentParts(blocks), {
        originalsDir: this.resolveOriginalsDir?.(this.sessionId),
      });
    } finally {
      this.pendingPromptAborts.delete(pending);
    }
    return pending.aborted ? undefined : content;
  }

  private async driveUnknownCommand(name: string): Promise<PromptResponse> {
    await this.conn.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: `Unknown ACP command: /${name}. Use /help to see available commands.`,
        },
      },
    });
    return { stopReason: 'end_turn' };
  }

  private driveSkillActivation(skillName: string, args: string): Promise<PromptResponse> {
    return this.enqueuePrompt(() => {
      this.assertNoActiveTurn();
      return this.driveLaunch(
        this.agent.activateSkill({ name: skillName, args: args.length > 0 ? args : undefined }),
      );
    });
  }

  private async driveBuiltinCommand(
    name: AcpBuiltinSlashCommandName,
    args: string,
    availableCommands: readonly AvailableCommand[],
  ): Promise<PromptResponse> {
    let text: string;
    try {
      text = await runBuiltinSlashCommand(
        name,
        {
          klient: this.klient,
          session: this.session,
          agent: this.agent,
          sessionId: this.sessionId,
          modelId: this.currentModelId,
          thinkingEnabled: this.currentThinkingLevel !== 'off',
          modeId: this.currentModeId,
          availableCommands,
        },
        args,
      );
    } catch (error) {
      log.warn('acp: builtin slash command failed', {
        sessionId: this.sessionId,
        command: name,
        error: error instanceof Error ? error.message : String(error),
      });
      text = `/${name} failed: ${error instanceof Error ? error.message : String(error)}`;
    }
    await this.conn.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text },
      },
    });
    return { stopReason: 'end_turn' };
  }

  private skillCommandMap(): ReadonlyMap<string, string> {
    return buildAcpSkillSlashCommands(this.skills).commandMap;
  }

  private assertNoActiveTurn(): void {
    if (this.driver !== undefined && !this.driver.settled) {
      throw RequestError.invalidRequest(
        { code: TURN_AGENT_BUSY_CODE },
        'another turn is already in progress',
      );
    }
  }

  private driveTurn(input: readonly ContentPart[]): Promise<PromptResponse> {
    this.assertNoActiveTurn();
    const promptId = `acp-${randomUUID()}`;
    return this.driveLaunch(this.agent.prompt({ input, promptId }), promptId);
  }

  private driveLaunch(launch: Promise<PromptLaunchResult>, promptId?: string): Promise<PromptResponse> {
    return new Promise<PromptResponse>((resolve, reject) => {
      const driver: TurnDriver = {
        resolve,
        reject,
        promptId,
        settled: false,
        launchPending: true,
        queued: false,
        awaitingResponseTurn: false,
        early: [],
      };
      this.driver = driver;
      launch.then(
        (launched) => {
          if (driver.settled) return;
          driver.launchPending = false;
          if (launched === undefined) {
            driver.queued = true;
            if (driver.cancelRequested === true) {
              this.cancelQueuedPrompt(driver);
            }
            return;
          }
          driver.turnId = launched.turn_id;
          if (driver.cancelRequested === true) {
            void this.agent.cancel({ turnId: launched.turn_id }).catch((error: unknown) => {
              log.warn('acp: deferred cancel failed', {
                sessionId: this.sessionId,
                error: error instanceof Error ? error.message : String(error),
              });
            });
          }
          for (const dispatch of driver.early.splice(0)) {
            dispatch();
          }
        },
        (error: unknown) => {
          this.settleDriver(driver, () => {
            reject(mapPromptLaunchError(error, this.sessionId));
          });
        },
      );
    });
  }

  private cancelQueuedPrompt(driver: TurnDriver): void {
    if (driver.promptId === undefined) return;
    void this.agent.cancelPrompt({ promptId: driver.promptId }).catch((error: unknown) => {
      log.warn('acp: queued-prompt cancel failed', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  private dispatchTurnEvent(dispatch: () => void): void {
    const driver = this.driver;
    if (
      driver !== undefined &&
      !driver.settled &&
      driver.turnId === undefined &&
      driver.launchPending
    ) {
      driver.early.push(dispatch);
      return;
    }
    dispatch();
  }

  private settleDriver(driver: TurnDriver, action: () => void): void {
    if (driver.settled) return;
    driver.settled = true;
    if (this.driver === driver) this.driver = undefined;
    action();
  }

  private driverFor(turnId: number): TurnDriver | undefined {
    const driver = this.driver;
    if (driver === undefined || driver.turnId === undefined || driver.turnId !== turnId) {
      return undefined;
    }
    return driver;
  }

  private onAssistantDelta(event: AgentEventPayloads['assistant.delta']): void {
    this.emitTurn(event.turnId, assistantDeltaToSessionUpdate(this.sessionId, event));
  }

  private onThinkingDelta(event: AgentEventPayloads['thinking.delta']): void {
    this.emitTurn(event.turnId, thinkingDeltaToSessionUpdate(this.sessionId, event));
  }

  private onToolCallStarted(event: AgentEventPayloads['tool.call.started']): void {
    const mapped = event as unknown as ToolCallStartedEvent;
    const key = acpToolCallId(event.turnId, event.toolCallId);
    const locations = toolCallLocations(mapped.name, mapped.args, mapped.display);
    if (locations !== undefined) {
      this.toolLocations.set(key, locations);
    }
    const diff = mapped.display === undefined ? null : displayBlockToAcpContent(mapped.display);
    if (diff?.type === 'diff') {
      this.toolDiffs.set(key, diff);
    }
    if (mapped.name === 'Bash') {
      const command = (mapped.args as { command?: unknown } | undefined)?.command;
      if (typeof command === 'string') {
        this.bashCallsAwaitingTerminal.set(key, command);
      }
    }
    const streamArgs = { args: stringifyArgs(mapped.args) };
    const lazyCreated = this.toolCallStreamArgs.has(key);
    this.toolCallStreamArgs.set(key, streamArgs);
    this.emitTurn(
      event.turnId,
      lazyCreated
        ? toolCallStartedUpgradeToSessionUpdate(this.sessionId, mapped)
        : toolCallStartToSessionUpdate(this.sessionId, mapped),
    );
    if (event.display !== undefined) {
      this.emitTurn(
        event.turnId,
        planFromDisplayBlock(this.sessionId, event.turnId, event.display as ToolInputDisplay),
      );
    }
  }

  private onToolCallDelta(event: AgentEventPayloads['tool.call.delta']): void {
    const mapped = event as unknown as ToolCallDeltaEvent;
    const key = acpToolCallId(event.turnId, event.toolCallId);
    const acc = this.toolCallStreamArgs.get(key);
    if (acc === undefined) {
      this.toolCallStreamArgs.set(key, { args: event.argumentsPart ?? '' });
      this.emitTurn(event.turnId, toolCallLazyCreateToSessionUpdate(this.sessionId, mapped));
      return;
    }
    this.emitTurn(event.turnId, toolCallDeltaToSessionUpdate(this.sessionId, mapped, acc));
  }

  private onToolUpdated(event: AgentEventPayloads['tool.updated']): void {
    const key = acpToolCallId(event.turnId, event.toolCallId);
    const status = event.fields['status'];
    if (typeof status !== 'string' || status.length === 0 || this.toolStatuses.get(key) === status) return;
    this.toolStatuses.set(key, status);
    this.emitTurn(
      event.turnId,
      toolUpdatedToSessionUpdate(this.sessionId, event as unknown as ToolUpdatedEvent),
    );
  }

  private onToolResult(event: AgentEventPayloads['tool.result']): void {
    const key = acpToolCallId(event.turnId, event.toolCallId);
    this.toolStatuses.delete(key);
    const locations = this.toolLocations.get(key);
    this.toolLocations.delete(key);
    const diff = this.toolDiffs.get(key);
    this.toolDiffs.delete(key);
    this.bashCallsAwaitingTerminal.delete(key);
    this.toolCallStreamArgs.delete(key);
    const terminalId = this.terminalBackedCalls.get(key);
    this.terminalBackedCalls.delete(key);
    if (terminalId !== undefined) {
      this.emitTurn(event.turnId, {
        sessionId: this.sessionId,
        update: {
          sessionUpdate: 'tool_call_update',
          toolCallId: key,
          status: event.isError === true ? 'failed' : 'completed',
          content: [{ type: 'terminal', terminalId }],
          locations,
        },
      });
      return;
    }
    this.emitTurn(
      event.turnId,
      toolResultToSessionUpdate(this.sessionId, event as unknown as ToolResultEvent, locations, diff),
    );
  }

  private onTerminalCreated(event: AcpTerminalCreatedEvent): void {
    if (event.sessionId !== this.sessionId) return;
    for (const [key, command] of this.bashCallsAwaitingTerminal) {
      if (command.length === 0 || !event.shellCommand.endsWith(command)) continue;
      this.bashCallsAwaitingTerminal.delete(key);
      this.terminalBackedCalls.set(key, event.terminalId);
      this.emitTurn(Number(key.split(':')[0]), {
        sessionId: this.sessionId,
        update: {
          sessionUpdate: 'tool_call_update',
          toolCallId: key,
          content: [{ type: 'terminal', terminalId: event.terminalId }],
        },
      });
      return;
    }
  }

  private onCompactionStarted(event: AgentEventPayloads['compaction.started']): void {
    if (event.trigger !== 'auto') return;
    this.emitLocalChunk(
      event.instruction === undefined
        ? 'Compacting conversation context…'
        : `Compacting conversation context with instruction: ${event.instruction}`,
    );
  }

  private onCompactionCompleted(event: AgentEventPayloads['compaction.completed']): void {
    this.emitLocalChunk(formatCompactionCompleted(event.result));
  }

  private emitLocalChunk(text: string): void {
    this.emit({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text },
      },
    });
  }

  private emitAgentWork(turnId: number, state: 'working' | 'idle', stopReason?: string): void {
    this.emitTurn(turnId, {
      sessionId: this.sessionId,
      update: {
        sessionUpdate: 'session_info_update',
        _meta: { jaz: { stateUpdate: { state, stopReason, tasks: this.takenIn.splice(0) } } },
      },
    });
  }

  private closeWork(end: TurnEnded): void {
    if (this.work === undefined) return;
    this.work = undefined;
    const error = end.error as { readonly code: string; readonly message?: string } | undefined;
    this.emitAgentWork(end.turnId, 'idle', end.reason === 'failed' ? 'error' : turnEndReasonToStopReason(end.reason, error));
    if (end.stopReason !== 'prompt_only') void this.emitUsageUpdate(end.turnId);
  }

  private reportTask(update: Record<string, unknown>, info: TaskEvent['info']): void {
    if (info.detached !== true) return;
    this.emit({ sessionId: this.sessionId, update: { ...update, asyncTaskId: info.taskId } } as unknown as SessionNotification);
  }

  private onTurnStarted(event: AgentEventPayloads['turn.started']): void {
    const ended = this.lastEnd;
    this.lastEnd = undefined;
    const driver = this.driver;
    const pending = driver !== undefined && !driver.settled && driver.turnId === undefined;
    const queuedTurn = driver?.queued === true && driver.promptId !== undefined && event.promptId === driver.promptId;
    if (pending && (driver.awaitingResponseTurn || queuedTurn)) {
      driver.queued = false;
      driver.awaitingResponseTurn = false;
      driver.turnId = event.turnId;
      if (driver.cancelRequested === true) {
        void this.agent.cancel({ turnId: event.turnId }).catch((error: unknown) => {
          log.warn('acp: deferred cancel of an adopted turn failed', {
            sessionId: this.sessionId,
            error: error instanceof Error ? error.message : String(error),
          });
        });
      }
    }
    for (const waiter of this.steerWaiters) {
      if (waiter.settled || waiter.turnId !== undefined) continue;
      if (waiter.promptId !== undefined && event.promptId === waiter.promptId) {
        waiter.turnId = event.turnId;
        waiter.awaitingNextTurn = false;
        continue;
      }
      if (waiter.awaitingNextTurn) {
        waiter.turnId = event.turnId;
        waiter.awaitingNextTurn = false;
      }
    }
    const owned = driver !== undefined && !driver.settled && (driver.launchPending || driver.turnId === event.turnId);
    if (this.work !== undefined && this.work.owned !== owned && ended !== undefined) this.closeWork(ended);
    const origin = event.origin as { readonly kind?: unknown; readonly taskId?: unknown } | undefined;
    if (origin?.kind === 'task' && typeof origin.taskId === 'string') this.takenIn.push(origin.taskId);
    if (this.work === undefined) {
      this.work = { owned };
      this.emitAgentWork(event.turnId, 'working');
    }
  }

  private onPromptCompleted(event: AgentEventPayloads['prompt.completed']): void {
    this.settleSteerWaitersByPrompt(event.promptId, 'end_turn');
    const driver = this.driver;
    if (driver === undefined || driver.settled || driver.promptId === undefined) return;
    if (event.promptId !== driver.promptId) return;
    if (driver.turnId !== undefined || driver.awaitingResponseTurn) return;
    this.settleDriver(driver, () => {
      driver.resolve({ stopReason: driver.cancelRequested === true ? 'cancelled' : 'end_turn' });
    });
  }

  private settleSteerWaitersByPrompt(promptId: string, stopReason: string): void {
    for (const waiter of this.steerWaiters) {
      if (waiter.settled || waiter.promptId !== promptId) continue;
      if (waiter.turnId !== undefined || waiter.awaitingNextTurn) continue;
      waiter.settle(stopReason);
    }
  }

  private onPromptAborted(event: AgentEventPayloads['prompt.aborted']): void {
    this.settleSteerWaitersByPrompt(event.promptId, 'cancelled');
    const driver = this.driver;
    if (driver === undefined || driver.settled || driver.promptId === undefined) return;
    if (event.promptId !== driver.promptId) return;
    this.settleDriver(driver, () => {
      driver.resolve({ stopReason: 'cancelled' });
    });
  }

  private onTurnEnded(event: TurnEnded): void {
    this.lastEnd = event;
    const completed = event.reason === 'completed';
    const error = event.error as { readonly code: string; readonly message?: string } | undefined;
    const continues = completed && event.continuation === 'turn';
    for (const waiter of this.steerWaiters) {
      if (waiter.settled || waiter.turnId !== event.turnId) continue;
      if (continues) {
        waiter.turnId = undefined;
        waiter.awaitingNextTurn = true;
        continue;
      }
      waiter.settle(turnEndReasonToStopReason(event.reason, error));
    }
    const driver = this.driverFor(event.turnId);
    if (driver !== undefined && continues && driver.cancelRequested !== true) {
      driver.turnId = undefined;
      driver.awaitingResponseTurn = true;
      return;
    }
    if (!continues) this.finishTurn(event);
    if (driver === undefined) return;
    this.settleDriver(driver, () => {
      if (event.reason === 'failed' && isAuthError(error)) {
        driver.reject(RequestError.authRequired(undefined, error?.message));
        return;
      }
      driver.resolve({ stopReason: driver.cancelRequested === true ? 'cancelled' : turnEndReasonToStopReason(event.reason, error) });
    });
  }

  private finishTurn(end: TurnEnded): void {
    this.lastEnd = undefined;
    this.closeWork(end);
    const error = end.error as { readonly code: string; readonly message?: string } | undefined;
    const stopReason = turnEndReasonToStopReason(end.reason, error);
    for (const waiter of this.steerWaiters) {
      if (waiter.awaitingNextTurn) waiter.settle(stopReason);
    }
    const driver = this.driver;
    if (driver === undefined || driver.settled || !driver.awaitingResponseTurn) return;
    this.settleDriver(driver, () => {
      driver.resolve({ stopReason: driver.cancelRequested === true ? 'cancelled' : stopReason });
    });
  }

  private async emitUsageUpdate(turnId?: number): Promise<void> {
    try {
      const size = (await this.klient.global.kosong.listModels()).find(
        (item) => item.model === this.currentModelId,
      )?.max_context_size;
      if (size === undefined) return;
      const context = await this.agent.getContext();
      this.emitTurn(
        turnId,
        usageUpdateNotification(this.sessionId, context.tokenCount, size, await this.usageReport()),
      );
    } catch (error) {
      log.warn('acp: failed to push usage_update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async usageReport(): Promise<AcpUsageReport | undefined> {
    let total: TokenUsage | undefined;
    try {
      total = (await this.agent.getUsage()).total;
    } catch {
      return undefined;
    }
    if (total === undefined) return undefined;
    const usage =
      this.reportedUsage === undefined ? total : subtractUsage(total, this.reportedUsage);
    this.reportedUsage = total;
    const input = usage.inputOther + (usage.inputCacheRead ?? 0) + usage.inputCacheCreation;
    const output = usage.output;
    if (input === 0 && output === 0) return undefined;
    return {
      usageId: randomUUID(),
      usage: {
        inputTokens: input,
        outputTokens: output,
        totalTokens: grandTotal(usage) ?? input + output,
        ...(usage.inputCacheRead === undefined
          ? {}
          : { cacheReadInputTokens: usage.inputCacheRead }),
        cacheCreationInputTokens: usage.inputCacheCreation,
      },
    };
  }
  private onMetadataChanged(event: SessionEventPayloads['metadata.changed']): void {
    if (!event.changed.includes('title')) return;
    void this.emitSessionInfoUpdate();
  }

  private async emitSessionInfoUpdate(): Promise<void> {
    try {
      const meta = await this.session.get();
      this.emit(sessionInfoUpdateNotification(this.sessionId, meta.title ?? null));
    } catch (error) {
      log.warn('acp: failed to push session_info_update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private emit(notification: SessionNotification | null): void {
    if (notification === null) return;
    void this.conn.sessionUpdate(notification).catch((error: unknown) => {
      log.warn('acp: failed to push session/update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  private emitTurn(turnId: number | undefined, notification: SessionNotification | null): void {
    if (notification === null) return;
    if (turnId !== undefined) {
      const update = notification.update as { _meta?: Record<string, unknown> };
      update._meta = { ...update._meta, turnId };
    }
    this.emit(notification);
  }

  cancel(): void {
    for (const pending of this.pendingPromptAborts) {
      pending.aborted = true;
    }
    for (const marker of this.chainedPrompts) {
      marker.cancelled = true;
    }
    for (const waiter of this.steerWaiters) {
      if (waiter.promptId !== undefined && waiter.turnId === undefined && !waiter.awaitingNextTurn) {
        void this.agent.cancelPrompt({ promptId: waiter.promptId }).catch((error: unknown) => {
          log.warn('acp: steered-prompt cancel failed', {
            sessionId: this.sessionId,
            error: error instanceof Error ? error.message : String(error),
          });
        });
      }
      waiter.settle('cancelled');
    }
    const driver = this.driver;
    if (driver === undefined || driver.settled) return;
    driver.cancelRequested = true;
    const turnId = driver.turnId;
    if (turnId !== undefined) {
      void this.agent.cancel({ turnId }).catch((error: unknown) => {
        log.warn('acp: cancel failed', {
          sessionId: this.sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      });
      return;
    }
    if (driver.awaitingResponseTurn) return;
    if (driver.queued) {
      this.settleDriver(driver, () => {
        driver.resolve({ stopReason: 'cancelled' });
      });
      this.cancelQueuedPrompt(driver);
      return;
    }
    void this.agent.cancel().catch((error: unknown) => {
      log.warn('acp: cancel (unaddressed) failed', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  async configOptions(): Promise<SessionConfigOption[]> {
    const models = projectModelCatalog(await this.klient.global.kosong.listModels());
    return buildSessionConfigOptions(
      models,
      this.currentModelId,
      this.currentThinkingLevel,
      this.currentModeId,
    );
  }

  modeState(): SessionModeState {
    return { currentModeId: this.currentModeId, availableModes: [...ACP_MODES] };
  }

  async setModel(id: string): Promise<void> {
    const suffix = ',thinking';
    const hasSuffix = id.endsWith(suffix);
    const baseId = hasSuffix ? id.slice(0, -suffix.length) : id;
    await this.agent.setModel(baseId);
    this.currentModelId = baseId;
    if (hasSuffix) {
      const models = projectModelCatalog(await this.klient.global.kosong.listModels());
      const level = models.find((model) => model.id === baseId)?.defaultThinkingEffort ?? 'on';
      await this.agent.setThinking(level);
      this.currentThinkingLevel = level;
    }
    await this.emitConfigOptionUpdate();
  }

  async setThinking(value: string): Promise<boolean> {
    const models = projectModelCatalog(await this.klient.global.kosong.listModels());
    const entry = models.find((model) => model.id === this.currentModelId);
    const efforts = entry?.supportEfforts;
    const alwaysThinking = entry?.alwaysThinking === true;
    const allowed =
      efforts !== undefined
        ? alwaysThinking
          ? efforts
          : ['off', ...efforts]
        : alwaysThinking
          ? ['on']
          : ['off', 'on'];
    let level: string;
    if (allowed.includes(value)) {
      level = value;
    } else if (value === 'on' && efforts !== undefined) {
      level = entry?.defaultThinkingEffort ?? 'on';
    } else {
      return false;
    }
    await this.agent.setThinking(level);
    this.currentThinkingLevel = level;
    await this.emitConfigOptionUpdate();
    return true;
  }

  async setMode(id: AcpModeId): Promise<void> {
    const { plan, permission } = acpModeToToggles(id);
    if (plan) {
      await this.agent.enterPlan();
    } else {
      await this.agent.cancelPlan();
    }
    await this.agent.setPermission(permission);
    this.currentModeId = id;
    this.emit(currentModeUpdateNotification(this.sessionId, id));
    await this.emitConfigOptionUpdate();
  }

  private async emitConfigOptionUpdate(): Promise<void> {
    try {
      await this.conn.sessionUpdate(
        configOptionUpdateNotification(this.sessionId, await this.configOptions()),
      );
    } catch (error) {
      log.warn('acp: failed to push config_option_update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
}

function formatCompactionCompleted(result: {
  readonly compactedCount: number;
  readonly tokensBefore: number;
  readonly tokensAfter: number;
}): string {
  return [
    'Compaction completed.',
    `- Messages compacted: ${result.compactedCount.toLocaleString('en-US')}`,
    `- Tokens before: ${result.tokensBefore.toLocaleString('en-US')}`,
    `- Tokens after: ${result.tokensAfter.toLocaleString('en-US')}`,
  ].join('\n');
}
