import Link from "next/link";

import { StatusBadge } from "@/components/Badges";
import { listRuns } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function RunsPage() {
  const runs = await listRuns();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-fg">Research runs</h1>
          <p className="text-sm text-muted">
            {runs.length} run{runs.length === 1 ? "" : "s"} — newest first
          </p>
        </div>
      </div>

      {runs.length === 0 ? (
        <div className="card text-sm text-muted">
          No runs yet. Start one from{" "}
          <Link href="/new" className="text-muted underline hover:text-fg">
            New run
          </Link>
          .
        </div>
      ) : (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border bg-surface text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-4 py-3">Objective</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Papers</th>
                <th className="px-4 py-3">Avg AGI score</th>
                <th className="px-4 py-3">Updated</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr
                  key={r.request_id}
                  className="border-b border-border/60 last:border-0 hover:bg-accent-soft"
                >
                  <td className="px-4 py-3">
                    <Link
                      href={`/runs/${r.request_id}`}
                      className="font-medium text-fg hover:text-muted"
                    >
                      {r.research_objective}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="px-4 py-3 text-fg">{r.paper_count}</td>
                  <td className="px-4 py-3 text-fg">
                    {r.average_agi_score !== null ? `${r.average_agi_score}/100` : "—"}
                  </td>
                  <td className="px-4 py-3 text-muted">
                    {new Date(r.updated_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
