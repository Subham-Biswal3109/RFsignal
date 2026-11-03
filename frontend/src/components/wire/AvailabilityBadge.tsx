/**
 * AvailabilityBadge — renders AVAILABLE / OCCUPIED / UNCERTAIN with
 * colour-coded styling.
 *
 * Props
 * ─────
 * available:  true  → AVAILABLE (green)
 *             false → OCCUPIED  (red)
 *             null  → UNCERTAIN (amber)
 */
import React from "react";
import { cn } from "@/lib/utils";

interface AvailabilityBadgeProps {
  available: boolean | null | undefined;
  /** Override the displayed text (defaults to AVAILABLE/OCCUPIED/UNCERTAIN). */
  label?: string;
  className?: string;
}

export function AvailabilityBadge({
  available,
  label,
  className,
}: AvailabilityBadgeProps) {
  const text =
    label ??
    (available === true
      ? "AVAILABLE"
      : available === false
      ? "OCCUPIED"
      : "UNCERTAIN");

  const colour =
    available === true
      ? "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-900/30 dark:text-emerald-300"
      : available === false
      ? "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-300"
      : "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-300";

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider",
        colour,
        className,
      )}
    >
      {text}
    </span>
  );
}
