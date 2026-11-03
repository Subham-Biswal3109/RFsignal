/**
 * Label — minimal shadcn-compatible Label component.
 */
import React from "react";
import { cn } from "@/lib/utils";

export type LabelProps = React.LabelHTMLAttributes<HTMLLabelElement>;

export function Label({ className, children, ...props }: LabelProps) {
  return (
    <label
      className={cn(
        "text-xs font-semibold uppercase tracking-wider text-gray-600",
        className,
      )}
      {...props}
    >
      {children}
    </label>
  );
}
