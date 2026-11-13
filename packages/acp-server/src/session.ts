/**
 * ACP session (v2) — drives a single main agent over one ACP `sessionId`,
 * through the `Klient` facade. `start.ts` creates the klient over the
 * in-memory transport; everything below goes through facade calls and typed
 * klient events, so swapping the transport (http / ipc) requires no change
 * here.
 *
 * `prompt` submits the user input via `agent.prompt(...)` and translates the
 * agent's scoped event stream — subscribed once per session in `init()`,
 * before the first prompt — into ACP `session/update` notifications via the
 * helpers in `./events-map`. The promise settles on the `turn.ended` event.
 * A submission the engine QUEUES behind another turn (the launch resolves
 * `undefined`) stays open and is tracked by its client-chosen `promptId`:
 * `turn.started` adopts the consuming turn once the queue drains, a
 * `prompt_only` flush rolls into the follow-up turn that actually answers,
 * and `prompt.completed` settles only a hook-blocked prompt that never runs.
 * Turns with no ACP owner (engine-internal notification turns) still stream
 * their events to the client — the agent's work stays visible between
 * prompts, matching the CLI.
 *
 * KLIENT GAPS (all reported; each marked `KLIENT-GAP` inline):
 *  - no session MCP connection view / compaction service → the `/mcp` and
 *    `/compact` builtin slash commands answer with an explanatory notice
 *    instead of live data (see `./builtin-commands`).
 *  - no `Turn.result` promise → settlement relies solely on `turn.ended`.
 */

import { randomUUID } from 'node:crypto';

import type {
  AvailableCommand,
  ContentBlock,
  PromptResponse,
  SessionConfigOption,
  SessionModeState,
  SessionNotification,
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
import { acpBlocksToContentParts, compressPromptImageParts } from './convert';
import {
  acpToolCallId,
  assistantDeltaToSessionUpdate,
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
  type AcpUsageTokens,
  isAuthError,
  stringifyArgs,
} from './events-map';
import { AcpInteractionBridge } from './interaction-bridge';
import { log } from './log';
import { projectModelCatalog } from './model-catalog';
import { ACP_MODES, type AcpModeId, acpModeToToggles, DEFAULT_MODE_ID } from './modes';
import { projectHistoryToSessionUpdates } from './replay';
import { buildAcpSkillSlashCommands, detectSlashIntent } from './slash';

/** Leading text of the first text block, if any (used for slash detection). */
function leadingText(blocks: readonly ContentBlock[]): string | undefined {
  const first = blocks[0];
  if (first !== undefined && first.type === 'text') return first.text;
  return undefined;
}

/**
 * The engine's wire code for "another turn is active". A plain
 * `agent.prompt` while busy is QUEUED by the engine (the RPC returns
 * `undefined`, indistinguishable from a hook-blocked launch), so this code
 * only reaches us from `agent.activateSkill`, which rejects instead.
 */
const TURN_AGENT_BUSY_CODE = 'turn.agent_busy';
const SESSION_OCCUPIED_CODE = 'session.occupied';
const SESSION_STALE_CODE = 'session.stale';

/**
 * Map a prompt-launch rejection (from `agent.prompt` / `agent.activateSkill`)
 * to the JSON-RPC error the client sees.
 *
 *  - Auth-coded failures surface as `auth_required` so the client drives its
 *    re-auth flow (same mapping as the `turn.ended` auth path).
 *  - `turn.agent_busy` maps to `invalidRequest` (-32600), matching the legacy
 *    adapter's busy-prompt semantics.
 *  - `session.occupied` / `session.stale` map to `invalidRequest` too:
 *    occupied passes the engine message through (it is user-facing already),
 *    stale swaps in a client-action message because the engine's names the
 *    internal session id.
 *  - Everything else becomes a fixed-message `internalError` (-32603): the
 *    raw engine message and stack are logged server-side but NEVER cross the
 *    wire, so internal details cannot leak into the JSON-RPC channel.
 */
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

/** Per-turn settlement state for one in-flight `session/prompt`. */
interface HostSlashCommandsSnapshot {
  readonly commands: ReadonlyArray<AvailableCommand>;
  readonly skillCommandMap?: ReadonlyMap<string, string>;
}

interface ResolvedCommands {
  readonly commands: AvailableCommand[];
  readonly skillCommandMap: ReadonlyMap<string, string>;
}

interface TurnDriver {
  resolve(response: PromptResponse): void;
  reject(error: unknown): void;
  /**
   * Client-chosen prompt record id handed to the engine
   * (`PromptPayload.promptId`), echoed on `turn.started` /
   * `prompt.completed` / `prompt.aborted` — the handle by which a QUEUED
   * prompt is tracked. Undefined on the skill-activation path (it never
   * queues; a busy engine rejects it instead).
   */
  promptId?: string;
  /**
   * Learned once the launch call (`agent.prompt` / `agent.activateSkill`)
   * resolves, or adopted from `turn.started` for a queued prompt. Events
   * carry a `turnId`, but until this is set no turn-scoped event can be
   * attributed to this prompt — and the turn may START (with a fast model
   * even END) before the launch round-trip delivers the id. Such events are
   * buffered in {@link early} instead of being dropped.
   */
  turnId?: number;
  settled: boolean;
  /**
   * Set when `cancel()` arrives while {@link turnId} is still unknown: the
   * launch handler re-issues a precisely-addressed cancel once the id lands,
   * and a no-launch outcome settles `cancelled` instead of `end_turn`.
   */
  cancelRequested?: boolean;
  /**
   * True while the launch round-trip is in flight: the prompt's turn may
   * already be streaming under an id we do not know yet, so turn-scoped
   * events buffer in {@link early}. False from the moment the launch
   * resolves — including the queued outcome, where the wait for
   * `turn.started` is event-driven and other turns' events must stream live.
   */
  launchPending: boolean;
  /**
   * The engine accepted the prompt but queued it behind another turn (the
   * launch resolved `undefined`). The driver stays open until events decide
   * it: `turn.started` with our promptId adopts the consuming turn,
   * `prompt.completed` settles a hook-blocked prompt that never runs,
   * `prompt.aborted` settles a cancelled one.
   */
  queued: boolean;
  /**
   * Our prompt's turn ended as `prompt_only`: the engine flushed it into
   * context without an LLM call because a notification turn was pending
   * behind it. That follow-up turn answers with our message in context, so
   * the driver stays open and adopts the next turn that starts.
   */
  awaitingResponseTurn: boolean;
  /**
   * Turn-scoped events that arrived while `turnId` was still unknown during
   * the launch round-trip. Once the launch resolves, every entry is replayed
   * in arrival order — the ones matching the driver's turn are its own
   * (possibly already finished) work, the rest are a still-draining prior
   * turn, which the live path would have streamed as ownerless traffic.
   */
  early: Array<{ readonly turnId: number; readonly dispatch: () => void }>;
}

export class AcpSession {
  /** The klient facade this session was created from. */
  private readonly klient: Klient;
  private readonly session: SessionHandle;
  private readonly agent: AgentHandle;

  /** Currently-selected model id (bare, no suffix). Empty when unbound. */
  private currentModelId: string = '';
  /** The engine's current thinking level verbatim (`'off'`, `'on'`, or an effort). */
  private currentThinkingLevel: string = 'off';
  /** Current ACP mode. */
  private currentModeId: AcpModeId = DEFAULT_MODE_ID;
  /**
   * Cached session skill summaries — the backing data for slash-intent
   * detection and `availableCommands()`. Seeded in `init()` and refreshed on
   * the klient `skills.changed` event.
   */
  private skills: readonly SkillSummary[] = [];
  /** The in-flight prompt's driver, if any. */
  private driver: TurnDriver | undefined;
  /**
   * FIFO chain serializing model-bound prompt submissions: each waits for
   * the previous prompt's driver to fully settle before launching, so a
   * client can type ahead while a turn runs instead of eating
   * `turn.agent_busy` rejections. Markers let `cancel()` / `dispose()`
   * settle still-chained prompts as `cancelled` without launching them.
   */
  private promptChain: Promise<unknown> = Promise.resolve();
  private readonly chainedPrompts = new Set<{ cancelled: boolean }>();
  /**
   * Abort markers of prompts still in their pre-turn image-compression phase
   * (no turn launched yet, so `agent.cancel` has nothing to cancel).
   * `cancel()` flips every marker; the prompt settles with
   * `stopReason: 'cancelled'` once compression finishes instead of launching
   * a turn the client already asked to stop.
   */
  private readonly pendingPromptAborts = new Set<{ aborted: boolean }>();
  /** Session-level agent-event subscriptions, torn down by `dispose()`. */
  private readonly subscriptions: IDisposable[] = [];
  /**
   * Streaming-args accumulators for in-flight tool calls, keyed by the ACP
   * wire toolCallId. An entry's existence doubles as "the wire `tool_call`
   * CREATE was sent" — either lazy-created from the first `tool.call.delta`
   * (the engine streams args deltas BEFORE `tool.call.started`) or created by
   * `tool.call.started` itself. Seeded/reseeded with the full stringified
   * args at started so any post-started delta emits cumulative REPLACE
   * content that includes the initial args. Cleaned at `tool.result`.
   */
  private readonly toolCallStreamArgs = new Map<string, { args: string }>();
  private readonly toolStatuses = new Map<string, string>();
  /**
   * Locations derived at `tool.call.started`, keyed by the ACP wire
   * toolCallId, re-attached to the terminal `tool_call_update`
   * (`tool.result` carries no args/display of its own).
   */
  private readonly toolLocations = new Map<string, ToolCallLocation[]>();
  /**
   * In-flight Bash tool calls awaiting terminal correlation, keyed by the ACP
   * wire toolCallId; the value is the model's `args.command`. Filled at
   * `tool.call.started`, consumed by {@link onTerminalCreated} (or dropped at
   * `tool.result`).
   */
  private readonly bashCallsAwaitingTerminal = new Map<string, string>();
  /**
   * Tool calls whose execution runs in a client terminal: ACP wire toolCallId
   * → terminalId. Their terminal `tool_call_update` carries a
   * `{type: 'terminal'}` content entry instead of the textual output (the
   * client already renders the bytes in the terminal pane — showing them in
   * the card too would duplicate them). The model still receives the full
   * captured output; only the client-facing card content is de-duplicated.
   */
  private readonly terminalBackedCalls = new Map<string, string>();
  /** Bridges engine approval / ask-user requests to the ACP client. */
  private readonly interactionBridge: AcpInteractionBridge;

  constructor(
    private readonly conn: AcpClient,
    klient: Klient,
    readonly sessionId: string,
    private readonly acpConnection: IAcpConnection,
    /**
     * Whether the client advertised `elicitation.form` at `initialize` —
     * forwarded to the interaction bridge's ask-user routing.
     */
    elicitationForm: boolean,
    /**
     * Resolve the session's media-originals dir for prompt-image compression
     * (`sessionMediaOriginalsDir(sessionDir)` when the live session scope is
     * reachable). Undefined / returning undefined → `persistOriginalImage`'s
     * shared temp-dir fallback applies.
     */
    private readonly resolveOriginalsDir?: (sessionId: string) => string | undefined,
    private readonly hostCommands:
      | ReadonlyArray<AvailableCommand>
      | HostSlashCommandsSnapshot = [],
  ) {
    this.klient = klient;
    this.session = klient.session(sessionId);
    // `main` is auto-materialized by the transport's scope resolution on the
    // first call — no explicit agent bootstrap is needed here.
    this.agent = this.session.agent('main');
    this.interactionBridge = new AcpInteractionBridge(conn, this.session, sessionId, elicitationForm);
  }

  /**
   * Subscribe the agent event stream and seed the config state. Must be
   * awaited before the first `prompt` so no early turn events are missed.
   */
  async init(): Promise<void> {
    const events = this.agent.events;
    this.subscriptions.push(
      events.on('assistant.delta', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onAssistantDelta(event);
        });
      }),
      events.on('thinking.delta', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onThinkingDelta(event);
        });
      }),
      events.on('tool.call.started', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onToolCallStarted(event);
        });
      }),
      events.on('tool.call.delta', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onToolCallDelta(event);
        });
      }),
      events.on('tool.updated', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onToolUpdated(event);
        });
      }),
      events.on('tool.result', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onToolResult(event);
        });
      }),
      events.on('turn.started', (event) => {
        this.onTurnStarted(event);
      }),
      events.on('turn.ended', (event) => {
        this.dispatchTurnEvent(event.turnId, () => {
          this.onTurnEnded(event);
        });
      }),
      events.on('prompt.completed', (event) => {
        this.onPromptCompleted(event);
      }),
      events.on('prompt.aborted', (event) => {
        this.onPromptAborted(event);
      }),
      // Compaction runs as a background LLM task outside any turn, so these
      // are not turn-scoped; the subscription is already agent-grained (this
      // session's main agent), which keeps other sessions' events out.
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
    // Session-scope stream: title changes surface as `session_info_update`.
    this.subscriptions.push(
      this.session.events.on('metadata.changed', (event) => {
        this.onMetadataChanged(event);
      }),
    );
    // Skill catalog changes refresh the cache and re-push the available
    // commands so the client's slash menu tracks the live catalog.
    this.subscriptions.push(
      this.session.events.on('skills.changed', () => {
        void this.refreshSkills().then(() => this.emitAvailableCommandsUpdate());
      }),
    );
    // Terminal correlation: the ACP-backed process runner (same process,
    // App-scope holder) announces every terminal it creates so this session
    // can attach it to the matching in-flight Bash tool call.
    const unsubscribeTerminal = this.acpConnection.onTerminalCreated((event) => {
      this.onTerminalCreated(event);
    });
    this.subscriptions.push({ dispose: unsubscribeTerminal });
    try {
      this.currentModelId = await this.agent.getModel();
      this.currentThinkingLevel = await this.agent.getThinking();
    } catch (error) {
      // Keep the unbound defaults — configOptions stays honest.
      log.warn('acp: could not seed model/thinking state', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    // Awaited: the post-`session/new` `available_commands_update` must already
    // carry the skills (see `activateSession`).
    await this.refreshSkills();
  }

  /** Refresh the skill cache from the session catalog (best-effort). */
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

  /**
   * Tear down per-session resources. Settles an in-flight prompt as
   * cancelled, stops forwarding approval / ask-user requests to the client,
   * and detaches the event subscriptions. Idempotent.
   */
  dispose(): void {
    this.cancel();
    const driver = this.driver;
    if (driver !== undefined) {
      // Never leave the JSON-RPC `session/prompt` hanging after teardown.
      this.settleDriver(driver, () => {
        driver.resolve({ stopReason: 'cancelled' });
      });
    }
    this.interactionBridge.dispose();
    for (const subscription of this.subscriptions.splice(0)) {
      subscription.dispose();
    }
  }

  /**
   * Replay the main agent's persisted context history as an ordered batch of
   * `session/update` notifications. Used by `session/load` so the client
   * re-renders prior turns before the response settles. Awaits every push for
   * ordering — replay is a one-shot batch, not a live stream.
   */
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
        // A single transient push failure must not truncate the whole replay;
        // log and continue so the rest of the history still lands.
        log.warn('acp: replayHistory failed to push a session/update; continuing', {
          sessionId: this.sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  /**
   * Resolve the current command catalog. Builtins always win, followed by
   * engine skills and then host-provided commands; duplicate names are dropped.
   * Host aliases only participate in skill activation when that alias is also
   * present in the advertised command list.
   */
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

  /** Return the same merged command palette that is advertised to the client. */
  availableCommands(): AvailableCommand[] {
    return this.resolveCommands().commands;
  }

  /** Push the current `available_commands_update` to the client. */
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
    // Slash-intent detection: builtin commands execute locally without an LLM
    // turn; skills activate through the engine (rendered skill prompt driven
    // as a normal turn); unknown slash commands are answered locally with an
    // "unknown command" notice (never sent to the model); non-slash input
    // goes to the model as-is.
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
      // Cancelled while compressing (see `cancel()`): settle without
      // launching a turn.
      return { stopReason: 'cancelled' };
    }
    return this.enqueuePrompt(() => this.driveTurn(content));
  }

  /**
   * Run a model-bound launch after every previously submitted prompt fully
   * settled. This is the client-visible queue: the engine may additionally
   * queue a submission behind its own internal turns (see {@link
   * TurnDriver.queued}), but two ACP prompts never overlap. Cancelled
   * markers settle as `cancelled` when the chain reaches them.
   */
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

  /**
   * Convert the ACP blocks to engine content parts and run the input-stage
   * image compression (see `compressPromptImageParts`). Compression happens
   * before any turn exists, so honor a `session/cancel` that arrives during
   * it: `cancel()` flips the marker and this returns `undefined` rather than
   * launching a turn the client already asked to stop. Returns the compressed
   * parts otherwise.
   */
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

  /**
   * Answer an unknown slash command locally (no LLM turn): push the notice as
   * one `agent_message_chunk`, then settle the prompt with `end_turn`. The
   * wording mirrors the legacy adapter's `runUnknownSlashCommand`.
   */
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

  /**
   * Activate a skill through the engine (the agent's `IAgentSkillService`
   * behind the klient facade): the engine renders the skill prompt (content + args)
   * and drives it as a normal turn, so the turn events stream and settle
   * exactly like a plain prompt. Empty args go over as `undefined`, matching
   * the other consumers.
   */
  private driveSkillActivation(skillName: string, args: string): Promise<PromptResponse> {
    return this.enqueuePrompt(() => {
      this.assertNoActiveTurn();
      return this.driveLaunch(
        this.agent.activateSkill({ name: skillName, args: args.length > 0 ? args : undefined }),
      );
    });
  }

  /**
   * Execute an ACP builtin slash command locally: render its text from live
   * klient/engine state (no LLM turn), push it as one `agent_message_chunk`,
   * then settle the prompt with `end_turn`. The chunk push is awaited so it
   * lands before the prompt response.
   */
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

  /**
   * The skill command lookup, projected from the cached skill summaries
   * (command name → skill name, including the `skill:`-prefixed entries).
   */
  private skillCommandMap(): ReadonlyMap<string, string> {
    return buildAcpSkillSlashCommands(this.skills).commandMap;
  }

  /**
   * Invariant guard: a model-bound launch must never overlap the in-flight
   * driver. {@link enqueuePrompt} serializes submissions, so this cannot
   * fire from the normal flow — it stays as the synchronous safety net
   * against a driver-assignment race, which would overwrite the in-flight
   * driver and leave the first prompt unsettled with both turns' events
   * unattributed (the failure the legacy adapter rejected with
   * `turn.agent_busy` → -32600).
   */
  private assertNoActiveTurn(): void {
    if (this.driver !== undefined && !this.driver.settled) {
      throw RequestError.invalidRequest(
        { code: TURN_AGENT_BUSY_CODE },
        'another turn is already in progress',
      );
    }
  }

  /**
   * Submit the prompt and drive the turn to completion: `agent.prompt()`
   * returns the launched turn id, which the session-level event handlers use
   * to attribute events to this driver. The promptId is client-chosen so a
   * QUEUED submission (busy engine → launch resolves `undefined`) can be
   * tracked to its consuming turn via `turn.started` (see {@link
   * onTurnStarted}). Settles on `turn.ended`.
   */
  private driveTurn(input: readonly ContentPart[]): Promise<PromptResponse> {
    this.assertNoActiveTurn();
    const promptId = `acp-${randomUUID()}`;
    return this.driveLaunch(this.agent.prompt({ input, promptId }), promptId);
  }

  /**
   * Shared turn settlement for every launch path (`agent.prompt`,
   * `agent.activateSkill`): a returned turn id attributes subsequent events
   * to this driver. `undefined` means the engine QUEUED the prompt behind
   * another turn or a submit hook blocked it — indistinguishable at this
   * point, so the driver stays open and the event stream decides
   * ({@link onTurnStarted} / {@link onPromptCompleted}).
   */
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
            // Queued or hook-blocked: keep the prompt open and wait for the
            // engine's events. A cancel that already arrived is re-issued
            // against the queued prompt precisely.
            driver.queued = true;
            if (driver.cancelRequested === true) {
              this.cancelQueuedPrompt(driver);
            }
            return;
          }
          driver.turnId = launched.turn_id;
          if (driver.cancelRequested === true) {
            // A cancel arrived before the id was known (see `cancel()`): the
            // unaddressed cancel may have predated the turn's activation, so
            // re-issue it now precisely addressed. Idempotent.
            void this.agent.cancel({ turnId: launched.turn_id }).catch((error) => {
              log.warn('acp: deferred cancel failed', {
                sessionId: this.sessionId,
                error: error instanceof Error ? error.message : String(error),
              });
            });
          }
          // Replay the events the turn emitted before its id arrived (a fast
          // turn can outrun the launch round-trip — `activateSkill` returns
          // only after the prompt-metadata update). Non-matching entries are
          // a still-draining prior turn; the live handlers stream those as
          // ownerless traffic, so replay everything in arrival order.
          for (const early of driver.early.splice(0)) {
            early.dispatch();
          }
        },
        (error) => {
          this.settleDriver(driver, () => {
            reject(mapPromptLaunchError(error, this.sessionId));
          });
        },
      );
    });
  }

  /**
   * Best-effort cancel of a prompt still sitting in the engine's queue. A
   * prompt that already launched rejects (`prompt.not_found`) — its
   * `turn.started` adoption re-cancels by turn id instead.
   */
  private cancelQueuedPrompt(driver: TurnDriver): void {
    if (driver.promptId === undefined) return;
    void this.agent.cancelPrompt({ promptId: driver.promptId }).catch((error) => {
      log.warn('acp: queued-prompt cancel failed', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  /**
   * Route a turn-scoped event: buffer it only while the launch round-trip is
   * in flight with the turn id still unknown (see {@link TurnDriver.early}).
   * Everything else dispatches live — including events of turns no ACP
   * prompt owns (engine-internal notification turns), which stream to the
   * client so the agent's work stays visible between prompts.
   */
  private dispatchTurnEvent(turnId: number, dispatch: () => void): void {
    const driver = this.driver;
    if (
      driver !== undefined &&
      !driver.settled &&
      driver.turnId === undefined &&
      driver.launchPending
    ) {
      driver.early.push({ turnId, dispatch });
      return;
    }
    dispatch();
  }

  /**
   * Settle the driver exactly once and detach it from the session so later
   * events of its turn are ignored.
   */
  private settleDriver(driver: TurnDriver, action: () => void): void {
    if (driver.settled) return;
    driver.settled = true;
    if (this.driver === driver) this.driver = undefined;
    action();
  }

  /** The active driver, but only for events of ITS turn. */
  private driverFor(turnId: number): TurnDriver | undefined {
    const driver = this.driver;
    if (driver === undefined || driver.turnId === undefined || driver.turnId !== turnId) {
      return undefined;
    }
    return driver;
  }

  private onAssistantDelta(event: AgentEventPayloads['assistant.delta']): void {
    this.emit(assistantDeltaToSessionUpdate(this.sessionId, event));
  }

  private onThinkingDelta(event: AgentEventPayloads['thinking.delta']): void {
    this.emit(thinkingDeltaToSessionUpdate(this.sessionId, event));
  }

  private onToolCallStarted(event: AgentEventPayloads['tool.call.started']): void {
    // The klient payload mirrors `ToolCallStartedEvent` (`args` / `display`
    // arrive as `unknown` — cast at this seam).
    const mapped = event as unknown as ToolCallStartedEvent;
    const key = acpToolCallId(event.turnId, event.toolCallId);
    const locations = toolCallLocations(mapped.name, mapped.args, mapped.display);
    if (locations !== undefined) {
      this.toolLocations.set(key, locations);
    }
    if (mapped.name === 'Bash') {
      const command = (mapped.args as { command?: unknown } | undefined)?.command;
      if (typeof command === 'string') {
        this.bashCallsAwaitingTerminal.set(key, command);
      }
    }
    // Branch on whether a streaming delta already lazy-created the wire
    // `tool_call` for this id:
    //  - YES → a second CREATE is illegal; emit the "upgrade"
    //    `tool_call_update` so title/kind/rawInput/locations (and any
    //    `display`-derived diff) land on the existing card and `status`
    //    flips to `'in_progress'`.
    //  - NO  → no prior deltas (provider doesn't stream args); emit the
    //    `tool_call` CREATE.
    // Either way the accumulator is (re)seeded with the full stringified
    // args: `tool_call_update` content is REPLACE-semantics, so a post-start
    // delta must emit the cumulative args string, not just its fragment.
    const streamArgs = { args: stringifyArgs(mapped.args) };
    const lazyCreated = this.toolCallStreamArgs.has(key);
    this.toolCallStreamArgs.set(key, streamArgs);
    this.emit(
      lazyCreated
        ? toolCallStartedUpgradeToSessionUpdate(this.sessionId, mapped)
        : toolCallStartToSessionUpdate(this.sessionId, mapped),
    );
    if (event.display !== undefined) {
      this.emit(
        planFromDisplayBlock(this.sessionId, event.turnId, event.display as ToolInputDisplay),
      );
    }
  }

  private onToolCallDelta(event: AgentEventPayloads['tool.call.delta']): void {
    // The klient payload mirrors `ToolCallDeltaEvent` field-for-field.
    const mapped = event as unknown as ToolCallDeltaEvent;
    const key = acpToolCallId(event.turnId, event.toolCallId);
    const acc = this.toolCallStreamArgs.get(key);
    if (acc === undefined) {
      // The engine emits args-stream deltas BEFORE `tool.call.started`
      // (deltas come from the provider's streaming phase; started is
      // dispatched when the call runs). Lazy-create the wire `tool_call`
      // from this first delta so subsequent updates have a legitimate parent
      // — clients otherwise surface "Tool call not found" until the start
      // eventually lands.
      this.toolCallStreamArgs.set(key, { args: event.argumentsPart ?? '' });
      this.emit(toolCallLazyCreateToSessionUpdate(this.sessionId, mapped));
      return;
    }
    // Subsequent delta — the helper accumulates the fragment and emits an
    // update with the cumulative args text (REPLACE-content semantics).
    this.emit(toolCallDeltaToSessionUpdate(this.sessionId, mapped, acc));
  }

  private onToolUpdated(event: AgentEventPayloads['tool.updated']): void {
    const key = acpToolCallId(event.turnId, event.toolCallId);
    const status = event.fields['status'];
    if (typeof status !== 'string' || status.length === 0 || this.toolStatuses.get(key) === status) return;
    this.toolStatuses.set(key, status);
    // The klient payload mirrors `ToolUpdatedEvent` field-for-field; the
    // helper forwards only `status` updates with text (as a title refresh)
    // and returns null for everything else, which `emit` drops.
    this.emit(toolUpdatedToSessionUpdate(this.sessionId, event as unknown as ToolUpdatedEvent));
  }

  private onToolResult(event: AgentEventPayloads['tool.result']): void {
    const key = acpToolCallId(event.turnId, event.toolCallId);
    this.toolStatuses.delete(key);
    const locations = this.toolLocations.get(key);
    this.toolLocations.delete(key);
    this.bashCallsAwaitingTerminal.delete(key);
    this.toolCallStreamArgs.delete(key);
    const terminalId = this.terminalBackedCalls.get(key);
    this.terminalBackedCalls.delete(key);
    if (terminalId !== undefined) {
      // Terminal-backed call: the client already renders the output bytes in
      // the terminal pane, so the card gets the terminal embed instead of a
      // textual copy. The full output still reached the model (and the
      // persisted wire record) untouched.
      this.emit({
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
    this.emit(
      toolResultToSessionUpdate(this.sessionId, event as unknown as ToolResultEvent, locations),
    );
  }

  /**
   * Correlate a freshly-created client terminal with the in-flight Bash tool
   * call whose command it runs, then attach a `{type: 'terminal'}` content
   * entry to that call's card. Match key: the runner reports the full shell
   * invocation (`cd <cwd> && <command>`), which ends with the model's
   * `args.command`. Terminals with no matching call (e.g. a subagent's —
   * this session only follows the main agent's events) stay unattached.
   */
  private onTerminalCreated(event: AcpTerminalCreatedEvent): void {
    if (event.sessionId !== this.sessionId) return;
    for (const [key, command] of this.bashCallsAwaitingTerminal) {
      if (command.length === 0 || !event.shellCommand.endsWith(command)) continue;
      this.bashCallsAwaitingTerminal.delete(key);
      this.terminalBackedCalls.set(key, event.terminalId);
      this.emit({
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

  /**
   * Report an auto-triggered compaction start. A manual `/compact` is already
   * acknowledged by the builtin command's reply chunk, so echoing its
   * `compaction.started` event too would double-report; an auto-triggered
   * compaction has no other client-visible signal.
   */
  private onCompactionStarted(event: AgentEventPayloads['compaction.started']): void {
    if (event.trigger !== 'auto') return;
    this.emitLocalChunk(
      event.instruction === undefined
        ? 'Compacting conversation context…'
        : `Compacting conversation context with instruction: ${event.instruction}`,
    );
  }

  /** Report the compaction result (token/message summary). */
  private onCompactionCompleted(event: AgentEventPayloads['compaction.completed']): void {
    this.emitLocalChunk(formatCompactionCompleted(event.result));
  }

  /** Push one local `agent_message_chunk` (best-effort, never throws). */
  private emitLocalChunk(text: string): void {
    this.emit({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text },
      },
    });
  }

  /**
   * Adopt turns for the waiting driver. Two cases: the queue drained to OUR
   * queued prompt (`turn.started` echoes its promptId), or the driver is
   * between a `prompt_only` flush and the follow-up turn that actually
   * answers (any `turn.started` — the engine schedules the pending
   * notification turn next, and it carries our message in context). The
   * event stream is ordered, so adoption lands before the turn's first
   * delta and every later event attributes through `driverFor`.
   */
  private onTurnStarted(event: AgentEventPayloads['turn.started']): void {
    const driver = this.driver;
    if (driver === undefined || driver.settled || driver.turnId !== undefined) return;
    if (driver.queued && driver.promptId !== undefined && event.promptId === driver.promptId) {
      driver.queued = false;
      driver.turnId = event.turnId;
      if (driver.cancelRequested === true) {
        // The queued-prompt cancel raced the launch (see `cancel()`): the
        // turn started anyway, so cancel it precisely addressed.
        void this.agent.cancel({ turnId: event.turnId }).catch((error) => {
          log.warn('acp: deferred cancel of an adopted turn failed', {
            sessionId: this.sessionId,
            error: error instanceof Error ? error.message : String(error),
          });
        });
      }
      return;
    }
    if (driver.awaitingResponseTurn) {
      driver.awaitingResponseTurn = false;
      driver.turnId = event.turnId;
    }
  }

  /**
   * Settle a driver whose prompt the engine finished without a turn. Only
   * the no-turn outcome settles here (a hook-blocked submission: the launch
   * resolved `undefined` and no `turn.started` followed). Once a turn was
   * adopted — or a `prompt_only` flush rolled the driver into the follow-up
   * turn — this event is bookkeeping and `turn.ended` governs settlement.
   */
  private onPromptCompleted(event: AgentEventPayloads['prompt.completed']): void {
    const driver = this.driver;
    if (driver === undefined || driver.settled || driver.promptId === undefined) return;
    if (event.promptId !== driver.promptId) return;
    if (driver.turnId !== undefined || driver.awaitingResponseTurn) return;
    this.settleDriver(driver, () => {
      driver.resolve({ stopReason: driver.cancelRequested === true ? 'cancelled' : 'end_turn' });
    });
  }

  /** Settle a driver whose queued prompt was cancelled before it ran. */
  private onPromptAborted(event: AgentEventPayloads['prompt.aborted']): void {
    const driver = this.driver;
    if (driver === undefined || driver.settled || driver.promptId === undefined) return;
    if (event.promptId !== driver.promptId) return;
    this.settleDriver(driver, () => {
      driver.resolve({ stopReason: 'cancelled' });
    });
  }

  private onTurnEnded(event: AgentEventPayloads['turn.ended']): void {
    const driver = this.driverFor(event.turnId);
    if (driver === undefined) return;
    if (event.stopReason === 'prompt_only') {
      // Context flush, not an answer: the engine appended the queued prompt
      // without an LLM call because a notification turn was pending behind
      // it. Keep the prompt open and adopt that follow-up turn when it
      // starts (see `onTurnStarted`) — its streamed work IS the response.
      driver.turnId = undefined;
      driver.awaitingResponseTurn = true;
      return;
    }
    const error = event.error as { readonly code: string; readonly message?: string } | undefined;
    this.settleDriver(driver, () => {
      // Auth failures must surface as a JSON-RPC `auth_required` error
      // so the client triggers its re-auth flow, not a silent `end_turn`.
      if (event.reason === 'failed' && isAuthError(error)) {
        driver.reject(RequestError.authRequired(undefined, error?.message));
        return;
      }
      driver.resolve({ stopReason: turnEndReasonToStopReason(event.reason, error) });
    });
    void this.emitUsageUpdate();
  }

  /**
   * Push a one-shot `usage_update` after a turn settles: `used` = the agent's
   * current context token count, `size` = the bound model's max context size
   * from the catalog. Skipped while no catalog model matches the bound id —
   * there is nothing honest to report. `cost` stays omitted (the engine has
   * no cost data). Recorded token totals ride `_meta.usage` (OpenAI/Codex
   * vocabulary) when the engine has any; a getUsage failure or an all-zero
   * tally falls back to the plain {used, size} shape.
   */
  private async emitUsageUpdate(): Promise<void> {
    try {
      const size = (await this.klient.global.kosong.listModels()).find(
        (item) => item.model === this.currentModelId,
      )?.max_context_size;
      if (size === undefined) return;
      const context = await this.agent.getContext();
      this.emit(
        usageUpdateNotification(this.sessionId, context.tokenCount, size, await this.tokenUsage()),
      );
    } catch (error) {
      log.warn('acp: failed to push usage_update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Token totals for the `usage_update` `_meta.usage` extension: the session's
   * cumulative `total` usage (falling back to the current turn's), mapped to
   * the OpenAI/Codex field names the host parses. Returns undefined when the
   * engine recorded no tokens at all (or getUsage failed) so the notification
   * stays in its legacy {used, size} shape.
   */
  private async tokenUsage(): Promise<AcpUsageTokens | undefined> {
    let usage: TokenUsage | undefined;
    try {
      const status = await this.agent.getUsage();
      usage = status.total ?? status.currentTurn;
    } catch {
      return undefined;
    }
    if (usage === undefined) return undefined;
    const input = usage.inputOther + (usage.inputCacheRead ?? 0) + usage.inputCacheCreation;
    const output = usage.output;
    if (input === 0 && output === 0) return undefined;
    return {
      inputTokens: input,
      outputTokens: output,
      totalTokens: grandTotal(usage) ?? input + output,
      ...(usage.inputCacheRead === undefined
        ? {}
        : { cacheReadInputTokens: usage.inputCacheRead }),
      cacheCreationInputTokens: usage.inputCacheCreation,
    };
  }
  private onMetadataChanged(event: SessionEventPayloads['metadata.changed']): void {
    if (!event.changed.includes('title')) return;
    void this.emitSessionInfoUpdate();
  }

  /** Push a `session_info_update` with the current title (best-effort). */
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

  /** Push a `session/update` notification (best-effort, never throws). */
  private emit(notification: SessionNotification | null): void {
    if (notification === null) return;
    void this.conn.sessionUpdate(notification).catch((error) => {
      log.warn('acp: failed to push session/update', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  /**
   * Cancel the in-flight turn, if any; abort every prompt still in its
   * pre-turn compression phase (those settle as cancelled without launching —
   * see {@link preparePromptContent}); and settle every prompt still waiting
   * in the local submission chain as cancelled. Idempotent.
   */
  cancel(): void {
    for (const pending of this.pendingPromptAborts) {
      pending.aborted = true;
    }
    for (const marker of this.chainedPrompts) {
      marker.cancelled = true;
    }
    const driver = this.driver;
    if (driver === undefined || driver.settled) return;
    driver.cancelRequested = true;
    const turnId = driver.turnId;
    if (turnId !== undefined) {
      void this.agent.cancel({ turnId }).catch((error) => {
        log.warn('acp: cancel failed', {
          sessionId: this.sessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      });
      return;
    }
    if (driver.awaitingResponseTurn) {
      // Between the prompt_only flush and the follow-up turn: nothing of
      // ours is running, so settle locally. The follow-up turn belongs to
      // the engine's queue, not to this prompt, and stays untouched.
      this.settleDriver(driver, () => {
        driver.resolve({ stopReason: 'cancelled' });
      });
      return;
    }
    if (driver.queued) {
      // Sitting in the engine's queue: settle now and drop the queued prompt
      // precisely. A racing `turn.started` re-cancels by turn id (see
      // `onTurnStarted`); a late `prompt.aborted` finds the driver settled.
      this.settleDriver(driver, () => {
        driver.resolve({ stopReason: 'cancelled' });
      });
      this.cancelQueuedPrompt(driver);
      return;
    }
    // The launch round-trip has not returned the turn id yet. The engine's
    // cancel payload makes turnId optional — an empty call cancels whatever
    // turn is active (the same contract kap-server's cancel route relies
    // on) — and concurrent prompts are rejected, so the active turn can only
    // be this driver's. Flag the driver too: when the id lands, the launch
    // handler re-issues a precisely-addressed cancel, and a no-launch
    // outcome settles `cancelled` instead of `end_turn`.
    void this.agent.cancel().catch((error) => {
      log.warn('acp: cancel (unaddressed) failed', {
        sessionId: this.sessionId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }

  /**
   * Build the current `configOptions` snapshot (model + thinking + mode).
   * The thinking toggle only appears when the bound model's catalog row is
   * thinking-capable (see `buildSessionConfigOptions`).
   */
  async configOptions(): Promise<SessionConfigOption[]> {
    const models = projectModelCatalog(await this.klient.global.kosong.listModels());
    return buildSessionConfigOptions(
      models,
      this.currentModelId,
      this.currentThinkingLevel,
      this.currentModeId,
    );
  }

  /**
   * The first-class `modes` ({@link SessionModeState}) snapshot for the
   * `session/new` / `session/load` / `session/resume` responses — the same
   * taxonomy the `mode` config-option arm projects.
   */
  modeState(): SessionModeState {
    return { currentModeId: this.currentModeId, availableModes: [...ACP_MODES] };
  }

  /**
   * Switch the active model.
   *
   * Legacy clients may merge the thinking flag into the model id as
   * `"<id>,thinking"` (mirrors the Python ref's `_ModelIDConv.from_acp_model_id`
   * and the legacy adapter): the merged form splits into `setModel(<bare id>)`
   * plus `setThinking(<the NEW model's default effort>)`. The asymmetry is
   * load-bearing — a bare id does NOT turn thinking off (model and thinking
   * stay orthogonal; disabling thinking requires the `thinking` config option
   * with value `'off'`). Both entry points (`session/set_model` and the
   * `model` arm of `session/set_config_option`) funnel here.
   */
  async setModel(id: string): Promise<void> {
    const suffix = ',thinking';
    const hasSuffix = id.endsWith(suffix);
    const baseId = hasSuffix ? id.slice(0, -suffix.length) : id;
    await this.agent.setModel(baseId);
    // Update BEFORE resolving the on-effort so a merged `,thinking` switch
    // picks the NEW model's default level, not the old one's.
    this.currentModelId = baseId;
    if (hasSuffix) {
      const models = projectModelCatalog(await this.klient.global.kosong.listModels());
      const level = models.find((model) => model.id === baseId)?.defaultThinkingEffort ?? 'on';
      await this.agent.setThinking(level);
      this.currentThinkingLevel = level;
    }
    await this.emitConfigOptionUpdate();
  }

  /**
   * Switch the thinking level. The value is validated against the current
   * model's declared capability: with `supportEfforts` the allowed set is
   * `'off'` plus every declared effort (an `always_thinking` model drops
   * `'off'`); without them it stays the boolean `'off'` / `'on'` pair. Legacy
   * boolean clients sending `'on'` to an effort-granular model map to the
   * model's default effort (`AcpModelEntry.defaultThinkingEffort`). Returns
   * `false` for a value the model cannot take — the caller maps that to ACP
   * `invalid_params`.
   */
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

  /** Switch the ACP mode (plan mode + permission mode). */
  async setMode(id: AcpModeId): Promise<void> {
    const { plan, permission } = acpModeToToggles(id);
    if (plan) {
      await this.agent.enterPlan();
    } else {
      // KLIENT-GAP(plan): `exitPlan` (`planService.exit()`) is not on the
      // klient surface; `cancelPlan` (`planModeCancel`) has the identical
      // state effect (see `agent/plan/planOps.ts`) — only the persisted op
      // name differs.
      await this.agent.cancelPlan();
    }
    await this.agent.setPermission(permission);
    this.currentModeId = id;
    // Both notifications fire: `current_mode_update` serves clients reading
    // the first-class `modes` state, `config_option_update` serves clients
    // reading the `mode` config-option arm. (Engine-side mode changes are not
    // observable — klient exposes no permission/plan change event.)
    this.emit(currentModeUpdateNotification(this.sessionId, id));
    await this.emitConfigOptionUpdate();
  }

  /** Push a fresh `config_option_update` to the client. */
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

/**
 * Render the client-facing summary of a finished compaction (mirrors the
 * legacy adapter's wording).
 */
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
