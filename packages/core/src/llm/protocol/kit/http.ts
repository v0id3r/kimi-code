/**
 * HTTP helpers for protocol clients: header normalization, request id lookup by header name, and
 * `Retry-After` parsing.
 */
import { headerValue } from '#/llm/endpoint';

export function headersToRecord(headers: unknown): Record<string, string> | null {
  if (
    headers === null ||
    typeof headers !== 'object' ||
    typeof (headers as { forEach?: unknown }).forEach !== 'function'
  ) {
    return null;
  }
  const record: Record<string, string> = {};
  (headers as { forEach(callback: (value: string, key: string) => void): void }).forEach(
    (value, key) => {
      record[key] = value;
    },
  );
  return record;
}

export function requestIdFromHeader(
  ...names: readonly string[]
): (headers: Readonly<Record<string, string>>) => string | undefined {
  return (headers) => {
    for (const name of names) {
      const value = headerValue(headers, name);
      if (value !== undefined) return value;
    }
    return undefined;
  };
}

export function parseRetryAfterMs(headers: unknown): number | null {
  const raw =
    headers !== null && typeof headers === 'object'
      ? typeof (headers as { get?: unknown }).get === 'function'
        ? (headers as { get(name: string): string | null }).get('retry-after')
        : headerValue(headers as Readonly<Record<string, string>>, 'retry-after')
      : null;
  if (raw === null || raw === undefined) return null;
  const seconds = Number.parseInt(raw, 10);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return seconds * 1000;
}
