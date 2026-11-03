/**
 * Panel — a simple card wrapper with optional title and subtitle.
 *
 * Used by PredictionForm, PredictionResultCard, and PredictionsTable to
 * provide consistent visual grouping.
 */
import React from "react";
import { cn } from "@/lib/utils";

interface PanelProps {
  title?: string;
  subtitle?: string;
  className?: string;
  children: React.ReactNode;
}

export function Panel({ title, subtitle, className, children }: PanelProps) {
  return (
    <div className={cn("panel rounded-xl border border-border bg-surface p-5 shadow-sm", className)}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
