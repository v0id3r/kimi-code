import type { LlmRecovery, LlmRecoveryRecord } from '#human/llm/policy/recovery';
import type { LlmResolver } from '#human/llm/policy/resolver';
import type { LlmRequest } from '#human/llm/request';
import type { IAgentMicroCompactionService } from '#/agent/microCompaction/microCompaction';

import type { AgentLoopProjector } from './projector';

export const STRUCTURE_RECOVERY_STRATEGY = 'structure';

export function hasStrictStructureRecovery(
  recoveries: readonly LlmRecoveryRecord[],
): boolean {
  return recoveries.some(
    (recovery) =>
      recovery.strategy === STRUCTURE_RECOVERY_STRATEGY && recovery.action === 'strict',
  );
}

/**
 * Micro-compaction runs first: it projects the cleared tool results into the
 * request, and (for turn requests) evaluates triggers and persists the newly
 * cleared outputs before the request is sent.
 */
export function createMicroCompactionResolver(
  microCompaction: IAgentMicroCompactionService,
  isTurnRequest: () => boolean,
): LlmResolver {
  return {
    id: 'micro-compaction',
    resolve: async (request) => {
      if (isTurnRequest()) {
        await microCompaction.prepareForTurnRequest();
      }
      if (!microCompaction.active()) return request;
      const messages = microCompaction.projectMessages(request.content.messages);
      return messages === request.content.messages
        ? request
        : { ...request, content: { ...request.content, messages } };
    },
  };
}

function projectRequest(
  request: LlmRequest,
  projector: AgentLoopProjector,
  strict: boolean,
): LlmRequest {
  return {
    ...request,
    content: {
      ...request.content,
      messages: projector.project(request.content.messages, strict),
    },
  };
}

export function createStructureResolver(projector: AgentLoopProjector): LlmResolver {
  return {
    id: 'structure',
    resolve: async (request, control) =>
      hasStrictStructureRecovery(control.appliedRecoveries)
        ? request
        : projectRequest(request, projector, false),
  };
}

export function createStructureRecovery(projector: AgentLoopProjector): LlmRecovery {
  return {
    propose: ({ error, request: recoveryRequest, appliedRecoveries }) => {
      if (error.kind !== 'request_structure') return undefined;
      if (hasStrictStructureRecovery(appliedRecoveries)) return undefined;
      return {
        strategy: STRUCTURE_RECOVERY_STRATEGY,
        action: 'strict',
        rewrite: (request) =>
          projectRequest(
            {
              ...request,
              content: { ...request.content, messages: recoveryRequest.content.messages },
            },
            projector,
            true,
          ),
      };
    },
  };
}
