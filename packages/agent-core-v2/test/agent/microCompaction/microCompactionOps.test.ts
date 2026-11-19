import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { DisposableStore } from '#/_base/di/lifecycle';
import { TestInstantiationService } from '#/_base/di/test';
import { SyncDescriptor } from '#/_base/di/descriptors';
import { IEventBus } from '#/app/event/eventBus';
import { EventBusService } from '#/app/event/eventBusService';
import { ContextApplyCompaction, ContextClear } from '#/agent/contextMemory/contextEvents';
import {
  microCompactionKey,
  MicroCompactionApply,
} from '#/agent/microCompaction/microCompactionOps';
import { UsageRecord } from '#/agent/usage/usageOps';
import { emptyUsage } from '#human/llm-requester/usage';
import { AppendLogStore } from '#/persistence/backends/node-fs/appendLogStore';
import { InMemoryStorageService } from '#/persistence/backends/memory/inMemoryStorageService';
import { IAppendLogStore } from '#/persistence/interface/appendLogStore';
import { IFileSystemStorageService } from '#/persistence/interface/storage';
import { IAgentStateService } from '#/agent/state/agentState';
import { IEventDispatcher } from '#/state/eventDispatcher';
import { AGENT_WIRE_RECORD_KEY, type WireRecord } from '#/wire/record';

import {
  registerTestAgentWire,
  registerTestEventDispatcher,
  testWireScope,
} from '../../wire/stubs';

const SCOPE = 'wire';
const KEY = 'micro-compaction-test';

let disposables: DisposableStore;
let dispatcher: IEventDispatcher;
let agentState: IAgentStateService;
let log: IAppendLogStore;

function buildHost(key: string): {
  dispatcher: IEventDispatcher;
  agentState: IAgentStateService;
  log: IAppendLogStore;
  eventBus: IEventBus;
} {
  const ix = disposables.add(new TestInstantiationService());
  ix.stub(IFileSystemStorageService, new InMemoryStorageService());
  ix.set(IAppendLogStore, new SyncDescriptor(AppendLogStore));
  ix.set(IEventBus, new SyncDescriptor(EventBusService));
  registerTestAgentWire(ix, testWireScope(SCOPE, key), {
    log: ix.get(IAppendLogStore),
    eventBus: ix.get(IEventBus),
  });
  const dispatcher = registerTestEventDispatcher(ix);
  ix.get(IAgentStateService).contributeState(microCompactionKey);
  return {
    dispatcher,
    agentState: ix.get(IAgentStateService),
    log: ix.get(IAppendLogStore),
    eventBus: ix.get(IEventBus),
  };
}

beforeEach(() => {
  disposables = new DisposableStore();
  const host = buildHost(KEY);
  dispatcher = host.dispatcher;
  agentState = host.agentState;
  log = host.log;
});

afterEach(() => disposables.dispose());

async function readRecords(key = KEY): Promise<WireRecord[]> {
  await dispatcher.flush();
  const out: WireRecord[] = [];
  for await (const record of log.read<WireRecord>(testWireScope(SCOPE, key), AGENT_WIRE_RECORD_KEY)) {
    out.push(record);
  }
  return out;
}

function applyEvent(cleared: readonly { toolCallId: string; path: string; tokens: number }[]) {
  return new MicroCompactionApply({
    agentId: 'test-agent',
    reason: 'threshold',
    clearedCount: cleared.length,
    freedTokens: cleared.reduce((sum, entry) => sum + entry.tokens, 0),
    boundary: 7,
    cleared: cleared.map((entry) => ({ ...entry })),
  });
}

describe('microCompaction ops (wire-backed)', () => {
  it('apply appends cleared entries and persists a micro_compaction.apply record', async () => {
    expect(agentState.get(microCompactionKey).cleared).toEqual([]);

    void dispatcher.dispatch(
      applyEvent([
        { toolCallId: 'c1', path: '/r/c1.txt', tokens: 3000 },
        { toolCallId: 'c2', path: '/r/c2.txt', tokens: 2000 },
      ]),
    );
    expect(agentState.get(microCompactionKey).cleared.map((entry) => entry.toolCallId)).toEqual([
      'c1',
      'c2',
    ]);

    const records = await readRecords();
    expect(records.map((record) => record.type)).toEqual(['micro_compaction.apply']);
    expect(records[0]).toEqual(
      expect.objectContaining({
        type: 'micro_compaction.apply',
        reason: 'threshold',
        clearedCount: 2,
        freedTokens: 5000,
        boundary: 7,
        agentId: 'test-agent',
      }),
    );
  });

  it('fold is idempotent per toolCallId (replay safety)', () => {
    void dispatcher.dispatch(applyEvent([{ toolCallId: 'c1', path: '/r/c1.txt', tokens: 3000 }]));
    const afterFirst = agentState.get(microCompactionKey);
    void dispatcher.dispatch(applyEvent([{ toolCallId: 'c1', path: '/r/c1.txt', tokens: 3000 }]));
    const afterDuplicate = agentState.get(microCompactionKey);
    expect(afterDuplicate).toBe(afterFirst);
    expect(afterDuplicate.cleared).toHaveLength(1);
  });

  it('resets after full compaction (context.apply_compaction)', () => {
    void dispatcher.dispatch(applyEvent([{ toolCallId: 'c1', path: '/r/c1.txt', tokens: 3000 }]));
    expect(agentState.get(microCompactionKey).cleared).toHaveLength(1);
    void dispatcher.dispatch(
      new ContextApplyCompaction({
        agentId: 'test-agent',
        summary: 'summary',
        compactedCount: 10,
      }),
    );
    expect(agentState.get(microCompactionKey).cleared).toEqual([]);
  });

  it('resets on context.clear', () => {
    void dispatcher.dispatch(applyEvent([{ toolCallId: 'c1', path: '/r/c1.txt', tokens: 3000 }]));
    void dispatcher.dispatch(new ContextClear({ agentId: 'test-agent' }));
    expect(agentState.get(microCompactionKey).cleared).toEqual([]);
  });
});

describe('microCompaction idle clock (usage.record fold)', () => {
  const usage = emptyUsage();

  it('records the time of the last turn-scoped response', () => {
    void dispatcher.dispatch(
      new UsageRecord(
        { agentId: 'test-agent', model: 'm', usage, usageScope: 'turn' },
        12345,
      ),
    );
    expect(agentState.get(microCompactionKey).lastTurnResponseAt).toBe(12345);
    void dispatcher.dispatch(
      new UsageRecord(
        { agentId: 'test-agent', model: 'm', usage, usageScope: 'turn' },
        23456,
      ),
    );
    expect(agentState.get(microCompactionKey).lastTurnResponseAt).toBe(23456);
  });

  it('ignores non-turn (operation) responses', () => {
    void dispatcher.dispatch(
      new UsageRecord(
        { agentId: 'test-agent', model: 'm', usage, usageScope: 'session' },
        12345,
      ),
    );
    expect(agentState.get(microCompactionKey).lastTurnResponseAt).toBeNull();
  });

  it('resets the clock on full compaction', () => {
    void dispatcher.dispatch(
      new UsageRecord(
        { agentId: 'test-agent', model: 'm', usage, usageScope: 'turn' },
        12345,
      ),
    );
    expect(agentState.get(microCompactionKey).lastTurnResponseAt).toBe(12345);
    void dispatcher.dispatch(
      new ContextApplyCompaction({
        agentId: 'test-agent',
        summary: 'summary',
        compactedCount: 10,
      }),
    );
    expect(agentState.get(microCompactionKey).lastTurnResponseAt).toBeNull();
  });
});
