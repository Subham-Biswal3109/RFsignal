/**
 * cn() — className merge utility.
 *
 * Uses clsx (conditional classes) with tailwind-merge (deduplication of
 * conflicting Tailwind utility classes).
 */
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
