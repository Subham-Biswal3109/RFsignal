/**
 * Wire Watcher — Centralized API Client & Fetch Helper
 *
 * Implements robust response parsing and error handling for all backend API endpoints.
 * Enforces:
 *   - Centralized base URL / endpoint path handling
 *   - Content-Type header checks before JSON parsing
 *   - Meaningful error diagnostics for network failures, non-200 HTTP statuses, and non-JSON responses
 *   - Safety against empty response bodies ("Unexpected end of JSON input")
 */

// Base URL: empty string uses Vite dev server proxy (/api -> http://localhost:5000)
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function apiFetch<T = any>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  // Normalize endpoint URL
  const url = endpoint.startsWith("http")
    ? endpoint
    : `${API_BASE}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers: {
        "Accept": "application/json",
        ...(options?.body ? { "Content-Type": "application/json" } : {}),
        ...options?.headers,
      },
    });
  } catch (err: any) {
    throw new Error(
      `Failed to fetch from backend at ${url}. Ensure the Flask backend server is running on port 5000.`
    );
  }

  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();

  if (!response.ok) {
    let errorMessage = `API HTTP ${response.status} (${response.statusText})`;
    if (rawText.trim()) {
      if (contentType.includes("application/json")) {
        try {
          const errJson = JSON.parse(rawText);
          errorMessage = errJson.error || errJson.details || errJson.message || errorMessage;
        } catch (_) {
          errorMessage = `${errorMessage}: ${rawText.slice(0, 300)}`;
        }
      } else {
        errorMessage = `${errorMessage}: ${rawText.slice(0, 300)}`;
      }
    }
    throw new Error(errorMessage);
  }

  if (!rawText.trim()) {
    throw new Error(`API returned an empty response (HTTP ${response.status})`);
  }

  if (!contentType.includes("application/json")) {
    throw new Error(
      `API returned non-JSON response (Content-Type: ${contentType}): ${rawText.slice(0, 300)}`
    );
  }

  try {
    return JSON.parse(rawText) as T;
  } catch (err: any) {
    throw new Error(`API returned invalid JSON: ${rawText.slice(0, 500)}`);
  }
}
