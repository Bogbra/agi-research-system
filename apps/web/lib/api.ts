import "server-only";

import type { RunDetail, RunSummary } from "./types";

/**
 * All backend calls happen here, and only here, on the server (Next.js
 * Server Components / Server Actions). The bearer token never reaches the
 * browser — the client only ever talks to this app's own server, which is
 * the point of the `server-only` import above: it makes accidentally
 * importing this module from a Client Component a build error, not a leaked
 * secret.
 */

const BACKEND_URL = process.env.BACKEND_API_URL ?? "http://localhost:8000";
const API_TOKEN = process.env.BACKEND_API_TOKEN ?? "dev-local-token";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/**
 * FastAPI error bodies are JSON — `{"detail": "..."}` for a raised
 * HTTPException, or `{"detail": [{"msg": "...", ...}, ...]}` for a Pydantic
 * validation error (422). Extract plain text from either shape so callers
 * never have to render a raw JSON blob as the error message.
 */
function extractErrorMessage(rawBody: string, fallback: string): string {
  if (!rawBody) return fallback;
  try {
    const parsed = JSON.parse(rawBody);
    const detail = parsed?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (typeof d?.msg === "string" ? d.msg : JSON.stringify(d)))
        .join("; ");
    }
  } catch {
    // Not JSON — fall through and show the raw body.
  }
  return rawBody;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${API_TOKEN}`,
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, extractErrorMessage(body, response.statusText));
  }
  if (response.status === 202 || response.status === 204) {
    return response.json().catch(() => undefined) as Promise<T>;
  }
  return response.json() as Promise<T>;
}

export function listRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/research");
}

export function getRun(requestId: string): Promise<RunDetail> {
  return request<RunDetail>(`/research/${encodeURIComponent(requestId)}`);
}

export function submitRun(objective: string): Promise<{ request_id: string; status: string }> {
  return request(`/research`, { method: "POST", body: JSON.stringify({ objective }) });
}
