# Jaz fork

This fork is upstream [kimi-code](https://github.com/MoonshotAI/kimi-code) 2.8.0 plus the patches
below. Each patch is either meant to go upstream or held as a temporary contract until the
upstream ACP surface covers it.

| Patch           | Where                                                              | What it changes                                                                                                                                       |
| --------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACP turn wiring | `packages/acp-server`, `packages/klient`                           | Steer waiters and queued prompt chaining settle through the `continuation` field on `turn.ended` (`'turn'` waits, anything else finishes the turn); per-turn usage slices, edit diffs in completed tool calls, detached background tasks with stop, and the work-state `stateUpdate` in `session_info_update` `_meta.jaz`. The klient contract exposes `task.started`/`task.terminated` and `stopReason` on `turn.ended`, which the core already emits. |
| Micro-compaction | `packages/agent-core-v2`, `packages/core`                         | Old tool results become compact placeholders in the request projection while the transcript keeps the full text; one-hour prompt cache TTL in the Kimi provider protocol. |
| Read host fix   | `packages/agent-core-v2/src/agent/tools/os/read/readTool.ts`       | A Read of a saved tool-result path resolves on the agent's host.                                                                                      |
| Jaz releases    | `.github/workflows/jaz-release.yml`                                | Release assets (native bundles) upload per platform for a dispatched version; bundles are built by the manual native bundle workflow.                  |

## Upstream tests

The acp-server suite carries the fork's cases (e2e turn lifecycle, steering, background tasks,
usage slices); agent-core-v2 carries the micro-compaction suites. Everything else runs as
upstream.

## 2.8.0 notes

Upstream 2.8.0 replaced the `loop.drained`/`loop.idle` lifecycle events with `continuation` on
`turn.ended`; the fork's settle path was re-keyed onto it (see above). Upstream 2.8.0 is still
ACP v1 only — there is no protocol v2 to move the temporary `_meta.jaz` work-state contract to.
