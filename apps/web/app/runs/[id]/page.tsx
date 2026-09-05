import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";

import { AutoRefresh } from "@/components/AutoRefresh";
import { ClassificationBadge, ScoreMeter, StatusBadge } from "@/components/Badges";
import { ApiError, getRun } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  let run;
  try {
    run = await getRun(id);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) notFound();
    throw error;
  }

  const isProcessing =
    run.status !== "completion" && run.status !== "failed" && run.status !== "evaluation_failed";
  const rankedPapers = [...run.evaluated_papers].sort(
    (a, b) => (b.agi_score ?? 0) - (a.agi_score ?? 0),
  );

  return (
    <div className="space-y-8">
      <AutoRefresh enabled={isProcessing} intervalMs={3000} />

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-fg">{run.research_objective}</h1>
          <p className="text-sm text-muted">{run.request_id}</p>
        </div>
        <StatusBadge status={run.status} />
      </div>

      {isProcessing && (
        <div className="card flex items-center gap-3 border-border bg-surface-elevated">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-subtle opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-subtle" />
          </span>
          <p className="text-sm text-fg">
            Run in progress ({run.status}) — this page updates automatically every few seconds.
          </p>
        </div>
      )}

      {run.error && (
        <div className="card border-red-500/30 bg-red-500/10 text-sm text-red-300">
          Run failed: {run.error}
        </div>
      )}

      {run.status === "evaluation_failed" && (
        <div className="card border-red-500/30 bg-red-500/10 text-sm text-red-300">
          Every discovered paper failed evaluation — this is not a normal completion that
          simply found nothing. See the failures below for details.
        </div>
      )}

      {run.evaluation_failures.length > 0 && (
        <div className="card border-amber-500/30 bg-amber-500/10 space-y-2">
          <h2 className="text-sm font-semibold text-amber-300">
            Evaluation failures ({run.evaluation_failures.length})
          </h2>
          <ul className="space-y-1 text-sm text-amber-200">
            {run.evaluation_failures.map((f) => (
              <li key={f.paper_id}>
                <span className="font-medium">{f.paper_title}</span> — {f.error_type} after{" "}
                {f.attempts} attempt{f.attempts === 1 ? "" : "s"}: {f.error_message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {run.errors.length > 0 && (
        <div className="card border-amber-500/30 bg-amber-500/10 space-y-1">
          <h2 className="text-sm font-semibold text-amber-300">Notes</h2>
          <ul className="list-inside list-disc space-y-0.5 text-sm text-amber-200">
            {run.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {run.average_agi_score !== null && (
        <div className="card space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h2 className="text-sm font-semibold text-fg">Average AGI-rubric score</h2>
            <ScoreMeter score={run.average_agi_score} />
          </div>
          <p className="text-sm text-muted">
            {run.paper_count} paper{run.paper_count === 1 ? "" : "s"} evaluated
          </p>
          <p className="text-xs text-subtle">
            Experimental abstract-level research-triage score; not an objective measure of AGI
            progress. See{" "}
            <a
              href="https://github.com/Bogbra/agi-research-system#what-the-score-means"
              target="_blank"
              rel="noreferrer"
              className="underline hover:text-muted"
            >
              what the score means
            </a>
            .
          </p>
        </div>
      )}

      {rankedPapers.length > 0 && (
        <div className="card space-y-4 p-0">
          <div className="border-b border-border p-5 pb-3">
            <h2 className="text-sm font-semibold text-fg">Ranked papers</h2>
          </div>
          <div className="space-y-0">
            {rankedPapers.map((paper, i) => (
              <div
                key={paper.link + i}
                className="border-b border-border/60 p-5 py-4 last:border-0"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <a
                      href={paper.link}
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium text-fg hover:underline"
                    >
                      {paper.title}
                    </a>
                    <p className="text-xs text-muted">{paper.authors.join(", ")}</p>
                  </div>
                  {paper.classification && (
                    <ClassificationBadge classification={paper.classification} />
                  )}
                </div>
                {paper.agi_score !== null && (
                  <div className="mt-2">
                    <ScoreMeter score={paper.agi_score} />
                  </div>
                )}
                <p className="mt-2 text-sm text-muted">{paper.overall_assessment}</p>
                {paper.key_innovations.length > 0 && (
                  <ul className="mt-2 list-inside list-disc space-y-0.5 text-sm text-fg">
                    {paper.key_innovations.map((innovation, j) => (
                      <li key={j}>{innovation}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {run.final_report && (
        <div className="card space-y-2">
          <h2 className="text-sm font-semibold text-fg">Full report</h2>
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{run.final_report}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
