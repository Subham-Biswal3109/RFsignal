/**
 * Formatting utilities used by the Wire Watcher frontend.
 *
 * All functions are pure and handle null/undefined gracefully by returning "—".
 */

/** Format a probability (0–1) as a percentage string, e.g. "67.3%". */
export function formatProbability(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

/** Format a frequency range, e.g. "119.975 – 120.025 MHz". */
export function formatFrequencyRange(
  start: number | null | undefined,
  end: number | null | undefined,
): string {
  if (start == null && end == null) return "—";
  if (start == null) return `— – ${end!.toFixed(3)} MHz`;
  if (end == null) return `${start.toFixed(3)} – — MHz`;
  return `${start.toFixed(3)} – ${end.toFixed(3)} MHz`;
}

/** Format a city + state location string. */
export function formatLocation(
  city: string | null | undefined,
  state: string | null | undefined,
): string {
  if (!city && !state) return "—";
  if (!city) return state!;
  if (!state) return city;
  return `${city}, ${state}`;
}

/**
 * Format a number with a fixed number of decimal places plus an optional unit.
 * Returns "—" for null/undefined/NaN/Infinity values.
 */
export function formatNumber(
  value: number | null | undefined,
  decimals = 1,
  unit = "",
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(decimals)}${unit}`;
}

/** Format an ISO-8601 timestamp as a locale date-time string. */
export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}
