import { appendFile, mkdir, mkdtemp, readFile, readdir, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { getLiveSessionById, ISessionContext } from '@moonshot-ai/agent-core-v2';
import { expect, it } from 'vitest';

import { createTestClient } from './_helpers/acpClient';
import { writeFakeModelConfig } from './_helpers/fakeModelConfig';
import { createScriptedProvider } from './_helpers/scriptedProvider';

it('restores a compacted tool result with Read when the ACP client only serves the workspace', async () => {
  const homeDir = await realpath(await mkdtemp(join(tmpdir(), 'acp-tool-result-read-')));
  const cwd = join(homeDir, 'work');
  await mkdir(cwd);
  await writeFakeModelConfig(homeDir);
  await appendFile(join(homeDir, 'config.toml'), `
[micro_compaction]
keep_recent = 0
min_result_tokens = 2
trigger_tokens = 0
clear_at_least_tokens = 1
`);
  const source = join(cwd, 'result.txt');
  await writeFile(source, 'first line\nrestored result\n' + 'more output on another line\n'.repeat(60));
  const provider = createScriptedProvider();
  const client = await createTestClient({ homeDir, extraSeeds: [provider.seed] });
  try {
    await client.send('initialize', { protocolVersion: 1, clientCapabilities: { fs: { readTextFile: true } } });
    client.onRequest('session/request_permission', () => ({ outcome: { outcome: 'selected', optionId: 'approve_once' } }));
    client.onRequest('fs/read_text_file', async (params) => {
      const { path } = params as { path: string };
      if (path !== source) throw new Error('path escapes the allowed directory');
      return { content: await readFile(path, 'utf8') };
    });
    const { sessionId } = await client.send('session/new', { cwd, mcpServers: [] }) as { sessionId: string };
    const session = getLiveSessionById(client.server.core.accessor, sessionId)!;
    const toolResults = join(session.accessor.get(ISessionContext).sessionDir, 'agents', 'main', 'tool-results');

    async function read(id: string, path: string, nLines = 1): Promise<string> {
      provider.mockNextResponse({
        type: 'function', id, name: 'Read', arguments: JSON.stringify({ path, line_offset: 2, n_lines: nLines }),
      });
      provider.mockNextText('done');
      expect(await client.send('session/prompt', {
        sessionId, prompt: [{ type: 'text', text: 'Read the requested file' }],
      })).toMatchObject({ stopReason: 'end_turn' });
      const result = provider.callHistory().at(-1)?.findLast((message) => message.role === 'tool' && message.toolCallId === id);
      return JSON.stringify(result);
    }

    const compacted = await read('original', source, 62);
    const [saved] = await readdir(toolResults);
    const savedPath = join(toolResults, saved!);
    expect(compacted).toContain(JSON.stringify(savedPath).slice(1, -1));
    expect(await read('restore', savedPath)).toContain('more output on another line');

    const privateFile = join(homeDir, 'private.txt');
    await writeFile(privateFile, 'private line\nmust not be returned\n');
    const link = join(toolResults, 'outside.txt');
    await symlink(privateFile, link);
    expect(await read('link', link)).toContain('path escapes the allowed directory');
  } finally {
    await client.close();
    await rm(homeDir, { recursive: true, force: true });
  }
}, 30_000);
