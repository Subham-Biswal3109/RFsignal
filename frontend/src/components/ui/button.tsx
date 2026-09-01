/**
 * Button — minimal shadcn-compatible Button component.
 * Supports variant and className props used by PredictionForm.
 */
import React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "outline" | "ghost" | "destructive";
}

export function Button({ variant = "default", className, children, ...props }: ButtonProps) {
  const base =
    "inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none";

  const variants: Record<string, string> = {
    default:     "bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500",
    secondary:   "bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-400",
    outline:     "border border-gray-300 bg-white hover:bg-gray-50 focus:ring-gray-400",
    ghost:       "hover:bg-gray-100 focus:ring-gray-400",
    destructive: "bg-red-600 text-white hover:bg-red-700 focus:ring-red-500",
  };

  return (
    <button className={cn(base, variants[variant], className)} {...props}>
      {children}
    </button>
  );
}
