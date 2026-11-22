import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { DisposableStore, toDisposable, type IDisposable } from '#/_base/di/lifecycle';
import { TestInstantiationService } from '#/_base/di/test';
import { IEventBus } from '#/app/event/eventBus';
import { EventBusService } from '#/app/event/eventBusService';
import { AppendLogStore } from '#/persistence/backends/node-fs/appendLogStore';
import { InMemoryStorageService } from '#/persistence/backends/memory/inMemoryStorageService';
import { IAppendLogStore } from '#/persistence/interface/appendLogStore';
import { IAgentStateService } from '#/agent/state/agentState';
import { AgentStateService } from '#/agent/state/agentStateService';
import { IEventDispatcher } from '#/state/eventDispatcher';
import { IConfigService } from '#/app/config/config';
import { IBootstrapService } from '#/app/bootstrap/bootstrap';
import { IFileSystemStorageService } from '#/persistence/interface/storage';
import { ISessionTokenCountingService } from '#/session/tokenCounting/sessionTokenCounting';
import { IAgentLoopService } from '#/agent/loop/loop';
import { IAgentScopeContext, makeAgentScopeContext } from '#/agent/scopeContext/scopeContext';
import type { ContextMessage } from '#/agent/contextMemory/types';
import { SyncDescriptor } from '#/_base/di/descriptors';
import { ILogService } from '#/_base/log/log';
import { UsageRecord } from '#/agent/usage/usageOps';
import { emptyUsage } from '@moonshot-ai/core/llm/usage';
import { microCompactionKey } from '#/agent/microCompaction/microCompactionOps';
import { microCompactionMarker, PROMPT_CACHE_TTL_MS } from '#/agent/microCompaction/microCompactionCore';
import {
  AgentMicroCompactionService,
} from '#/agent/microCompaction/microCompactionService';
import { IAgentMicroCompactionService } from '#/agent/microCompaction/microCompaction';
import { AGENT_WIRE_RECORD_KEY, type WireRecord } from '#/wire/record';

import {
  registerTestAgentWire,
  registerTestEventDispatcher,
  testWireScope,
} from '../../wire/stubs';

const SCOPE = 'wire';
const KEY = 'micro-compaction-service-test';

interface RecordedWrite {
  scope: string;
  key: string;
  text: string;
}

function exchange(toolName: string, callId: string, text: string): ContextMessage[] {
  return [
    {
      role: 'assistant',
      content: [],
      toolCalls: [{ type: 'function', id: callId, name: toolName, arguments: null }],
    },
    { role: 'tool', content: [{ type: 'text', text }], toolCallId: callId },
  ];
}

function textOf(message: ContextMessage): string {
  return message.role === 'tool'
    ? message.content.map((part) => (part.type === 'text' ? part.text : '')).join('')
    : '';
}

class IdempotentAgentStateService extends AgentStateService {
  override contributeState(key: Parameters<AgentStateService['contributeState']>[0]): IDisposable {
    if (this.has(key)) return toDisposable(() => {});
    return super.contributeState(key);
  }
}

describe('AgentMicroCompactionService', () => {
  let disposables: DisposableStore;
  let writes: RecordedWrite[];
  let warnCalls: { message: string; fields: Record<string, unknown> }[];
  let failWrites: boolean;
  let history: ContextMessage[];
  let dispatcher: IEventDispatcher;
  let log: IAppendLogStore;
  let service: IAgentMicroCompactionService;

  function buildService(options: {
    history: ContextMessage[];
    contextSize: number;
    config: Record<string, unknown> | undefined;
    failWrites?: boolean;
  }): IAgentMicroCompactionService {
    const ix = disposables.add(new TestInstantiationService());
    writes = [];
    warnCalls = [];
    failWrites = options.failWrites ?? false;
    const memory = new InMemoryStorageService();
    ix.stub(
      IFileSystemStorageService,
      new Proxy(memory, {
        get(target, prop) {
          if (prop === 'write') {
            return async (scope: string, key: string, bytes: Uint8Array, options?: unknown) => {
              if (failWrites) throw new Error('disk full');
              writes.push({ scope, key, text: new TextDecoder().decode(bytes) });
              return (target as unknown as {
                write(scope: string, key: string, bytes: Uint8Array, options?: unknown): Promise<void>;
              }).write(scope, key, bytes, options);
            };
          }
          const value = Reflect.get(target, prop, target);
          return typeof value === 'function' ? value.bind(target) : value;
        },
      }),
    );
    ix.set(IAppendLogStore, new SyncDescriptor(AppendLogStore));
    ix.set(IEventBus, new SyncDescriptor(EventBusService));
    registerTestAgentWire(ix, testWireScope(SCOPE, KEY), {
      log: ix.get(IAppendLogStore),
      eventBus: ix.get(IEventBus),
    });
    ix.set(IAgentStateService, new IdempotentAgentStateService());
    ix.get(IAgentStateService).contributeState(microCompactionKey);
    dispatcher = registerTestEventDispatcher(ix);
    log = ix.get(IAppendLogStore);
    ix.stub(IAgentScopeContext, makeAgentScopeContext({ agentId: 'test-agent', agentScope: SCOPE }));
    ix.stub(IConfigService, {
      get: ((key: string) =>
        key === 'microCompaction' ? options.config : undefined) as IConfigService['get'],
    } as unknown as IConfigService);
    ix.stub(IBootstrapService, { homeDir: '/home/test' } as unknown as IBootstrapService);
    ix.stub(ILogService, {
      warn: (message: string, fields?: Record<string, unknown>) => {
        warnCalls.push({ message, fields: fields ?? {} });
      },
    } as unknown as ILogService);
    ix.stub(IAgentLoopService, { history: () => options.history } as unknown as IAgentLoopService);
    const size = options.contextSize;
    ix.stub(ISessionTokenCountingService, {
      _serviceBrand: undefined,
      strategy: 'estimate' as never,
      get: () => ({ size }) as never,
      measured: () => undefined,
      latestMeasured: () => size,
      statusSize: () => size,
      recordTruncation: () => undefined,
      rebase: () => undefined,
      requestSize: () => size,
      estimateText: (text: string) => text.length,
      estimateMessage: (message: ContextMessage) =>
        message.role === 'tool' ? textOf(message).length : 0,
      estimateMessages: (messages: readonly ContextMessage[]) =>
        messages.reduce((sum, message) => sum + (message.role === 'tool' ? textOf(message).length : 0), 0),
      estimateTools: () => 0,
    });
    return ix.createInstance(AgentMicroCompactionService);
  }

  beforeEach(() => {
    disposables = new DisposableStore();
  });

  afterEach(() => disposables.dispose());

  it('saves cleared outputs, writes a micro_compaction.apply record, and projects markers', async () => {
    const big = 'x'.repeat(4000);
    history = [
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      ...exchange('Bash', 'c1', big),
      ...exchange('Bash', 'c2', big),
      ...exchange('Bash', 'c3', big),
      ...exchange('Bash', 'c4', big),
      ...exchange('Bash', 'c5', big),
      ...exchange('Bash', 'c6', 'recent'),
    ];
    service = buildService({
      history,
      contextSize: 2000,
      config: {
        enabled: true,
        keepRecent: 2,
        minResultTokens: 100,
        triggerTokens: 1000,
        clearAtLeastTokens: 500,
        idleMarginSeconds: 300,
      },
    });

    await service.prepareForTurnRequest();

    expect(service.active()).toBe(true);
    // keepRecent=2 protects c5 and c6; c1..c4 clear.
    expect(writes.map((write) => write.key.replace(/-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.txt$/, ''))).toEqual([
      'Bash-c1',
      'Bash-c2',
      'Bash-c3',
      'Bash-c4',
    ]);
    expect(writes[0]!.text).toBe(big);
    expect(writes[0]!.scope).toContain('tool-results');

    await dispatcher.flush();
    const records: WireRecord[] = [];
    for await (const record of log.read<WireRecord>(testWireScope(SCOPE, KEY), AGENT_WIRE_RECORD_KEY)) {
      records.push(record);
    }
    const apply = records.find((record) => record.type === 'micro_compaction.apply');
    expect(apply).toBeDefined();
    expect(apply).toEqual(
      expect.objectContaining({
        reason: 'threshold',
        clearedCount: 4,
        boundary: expect.any(Number),
      }),
    );

    const projected = service.projectMessages(history) as ContextMessage[];
    const clearedMessage = projected.find(
      (message) => message.role === 'tool' && message.toolCallId === 'c1',
    );
    expect(clearedMessage?.content).toEqual([
      { type: 'text', text: microCompactionMarker('/home/test/' + writes[0]!.scope + '/' + writes[0]!.key) },
    ]);
    // Prefix before the first cleared result is untouched.
    const userIndex = history.findIndex((message) => message.role === 'user');
    expect(projected[userIndex]).toBe(history[userIndex]);
    // Recent results stay intact.
    const recent = projected.find(
      (message) => message.role === 'tool' && message.toolCallId === 'c5',
    );
    expect(recent?.content).toEqual([{ type: 'text', text: big }]);
  });

  it('does nothing when disabled', async () => {
    history = [...exchange('Bash', 'c1', 'x'.repeat(4000)), ...exchange('Bash', 'c2', 'tail')];
    service = buildService({ history, contextSize: 999999, config: { enabled: false } });
    await service.prepareForTurnRequest();
    expect(service.active()).toBe(false);
    expect(writes).toEqual([]);
  });

  it('fires with idle reason after a restored session idled past the cache TTL', async () => {
    const big = 'x'.repeat(4000);
    history = [
      ...exchange('Bash', 'c1', big),
      ...exchange('Bash', 'c2', big),
      ...exchange('Bash', 'c3', 'tail'),
    ];
    service = buildService({
      history,
      contextSize: 500,
      config: {
        enabled: true,
        keepRecent: 1,
        minResultTokens: 100,
        triggerTokens: 1000,
        clearAtLeastTokens: 300,
        idleMarginSeconds: 300,
      },
    });

    // Restored session: the last turn response came from replayed history,
    // long before the restart.
    const lastResponseAt = 1_000_000;
    void dispatcher.dispatch(
      new UsageRecord(
        { agentId: 'test-agent', model: 'm', usage: emptyUsage(), usageScope: 'turn' },
        lastResponseAt,
      ),
    );

    await service.prepareForTurnRequest(lastResponseAt + PROMPT_CACHE_TTL_MS + 300_000 + 1);

    await dispatcher.flush();
    const records: WireRecord[] = [];
    for await (const record of log.read<WireRecord>(testWireScope(SCOPE, KEY), AGENT_WIRE_RECORD_KEY)) {
      records.push(record);
    }
    const apply = records.find((record) => record.type === 'micro_compaction.apply');
    expect(apply).toEqual(
      expect.objectContaining({ reason: 'idle', clearedCount: 2, agentId: 'test-agent' }),
    );
  });

  it('logs the path and reason when saving fails, and keeps the result uncleared', async () => {
    history = [...exchange('Bash', 'c1', 'x'.repeat(4000)), ...exchange('Bash', 'c2', 'tail')];
    service = buildService({
      history,
      contextSize: 2000,
      failWrites: true,
      config: {
        enabled: true,
        keepRecent: 0,
        minResultTokens: 100,
        triggerTokens: 1000,
        clearAtLeastTokens: 300,
        idleMarginSeconds: 300,
      },
    });

    await service.prepareForTurnRequest();

    expect(writes).toEqual([]);
    expect(service.active()).toBe(false);
    expect(warnCalls).toHaveLength(1);
    expect(warnCalls[0]!.fields['toolCallId']).toBe('c1');
    expect(String(warnCalls[0]!.fields['path'])).toContain('tool-results');
    expect(String(warnCalls[0]!.fields['path'])).toContain('Bash-c1');
    expect(warnCalls[0]!.fields['error']).toBe('disk full');
  });
});
