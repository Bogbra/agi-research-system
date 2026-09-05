"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/**
 * Polls the current route by re-running its Server Components on an
 * interval. Used on the run detail page while a run is still in progress —
 * the pipeline executes in a FastAPI BackgroundTask with no push channel
 * back to the browser, so polling is the simplest way to make the page
 * reflect progress without a manual reload.
 */
export function AutoRefresh({
  enabled,
  intervalMs = 3000,
}: {
  enabled: boolean;
  intervalMs?: number;
}) {
  const router = useRouter();

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => router.refresh(), intervalMs);
    return () => clearInterval(id);
  }, [enabled, intervalMs, router]);

  return null;
}
