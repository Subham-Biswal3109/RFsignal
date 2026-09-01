/**
 * Input — High-contrast, sharp typography input component.
 */
import React from "react";
import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          "flex h-9 w-full rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-1.5 text-sm font-mono font-semibold text-slate-900 dark:text-slate-50 shadow-sm",
          "placeholder:text-slate-400 placeholder:font-sans placeholder:font-normal",
          "focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 dark:focus:ring-sky-400",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";
