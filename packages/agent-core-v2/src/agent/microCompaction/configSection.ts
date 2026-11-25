import { z } from 'zod';

import { registerConfigSection } from '#/app/config/configSectionContributions';

export const MICRO_COMPACTION_SECTION = 'microCompaction';

export const MicroCompactionSchema = z.object({
  enabled: z.boolean().optional(),
  keepRecent: z.number().int().min(0).optional(),
  minResultTokens: z.number().int().min(0).optional(),
  triggerTokens: z.number().int().min(0).optional(),
  clearAtLeastTokens: z.number().int().min(0).optional(),
  idleMarginSeconds: z.number().int().min(0).optional(),
});

export type MicroCompactionConfig = z.infer<typeof MicroCompactionSchema>;

export const MICRO_COMPACTION_DEFAULTS = {
  enabled: true,
  keepRecent: 5,
  minResultTokens: 2000,
  triggerTokens: 100000,
  clearAtLeastTokens: 30000,
  idleMarginSeconds: 300,
} as const;

export interface MicroCompactionSettings {
  readonly enabled: boolean;
  readonly keepRecent: number;
  readonly minResultTokens: number;
  readonly triggerTokens: number;
  readonly clearAtLeastTokens: number;
  readonly idleMarginSeconds: number;
}

export function resolveMicroCompactionSettings(
  config: MicroCompactionConfig | undefined,
): MicroCompactionSettings {
  return {
    enabled: config?.enabled ?? MICRO_COMPACTION_DEFAULTS.enabled,
    keepRecent: config?.keepRecent ?? MICRO_COMPACTION_DEFAULTS.keepRecent,
    minResultTokens: config?.minResultTokens ?? MICRO_COMPACTION_DEFAULTS.minResultTokens,
    triggerTokens: config?.triggerTokens ?? MICRO_COMPACTION_DEFAULTS.triggerTokens,
    clearAtLeastTokens:
      config?.clearAtLeastTokens ?? MICRO_COMPACTION_DEFAULTS.clearAtLeastTokens,
    idleMarginSeconds:
      config?.idleMarginSeconds ?? MICRO_COMPACTION_DEFAULTS.idleMarginSeconds,
  };
}

registerConfigSection(MICRO_COMPACTION_SECTION, MicroCompactionSchema);
