"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { submitRunAction } from "@/app/actions";

const EXAMPLE_OBJECTIVES = [
  "AGI progress in meta-learning from the last week",
  "Cross-domain task transfer papers from the last two weeks",
  "Few-shot learning breakthroughs from the last month",
];

export function NewRunForm() {
  const router = useRouter();
  const [objective, setObjective] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function submit() {
    setError(null);
    if (!objective.trim()) {
      setError("Enter a research objective.");
      return;
    }
    startTransition(async () => {
      const result = await submitRunAction(objective.trim());
      if (result.ok) {
        router.push(`/runs/${result.requestId}`);
      } else {
        setError(result.error);
      }
    });
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {EXAMPLE_OBJECTIVES.map((example) => (
          <button
            key={example}
            onClick={() => setObjective(example)}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-xs text-muted hover:border-border-strong hover:text-fg"
          >
            {example}
          </button>
        ))}
      </div>

      <textarea
        className="h-24 w-full rounded-lg border border-border bg-bg p-3 text-sm text-fg"
        value={objective}
        onChange={(e) => setObjective(e.target.value)}
        placeholder="e.g. AGI progress in meta-learning from the last week"
        spellCheck={false}
      />

      {error && <p className="text-sm text-red-300">{error}</p>}

      <button
        onClick={submit}
        disabled={isPending}
        className="rounded-lg bg-accent-fill px-4 py-2 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-50"
      >
        {isPending ? "Starting…" : "Start research run"}
      </button>
    </div>
  );
}
