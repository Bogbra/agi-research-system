import type { Classification, RunStatus } from "@/lib/types";

const STATUS_STYLES: Record<RunStatus, string> = {
  // In-progress phases — not a quality signal, so these use the shared
  // chrome palette rather than the semantic red/amber/green used below.
  initialization: "bg-secondary text-muted ring-border-strong",
  planning: "bg-gold/15 text-gold-hover ring-gold/30",
  discovery: "bg-gold/15 text-gold-hover ring-gold/30",
  evaluation: "bg-gold/15 text-gold-hover ring-gold/30",
  completion: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
  failed: "bg-red-500/15 text-red-300 ring-red-500/30",
};

const CLASSIFICATION_STYLES: Record<Classification, string> = {
  "High AGI Potential": "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
  "Medium AGI Potential": "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  "Low AGI Potential": "bg-secondary text-muted ring-border-strong",
};

function Badge({ className, children }: { className: string; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${className}`}
    >
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: RunStatus }) {
  return <Badge className={STATUS_STYLES[status]}>{status.replaceAll("_", " ")}</Badge>;
}

export function ClassificationBadge({ classification }: { classification: Classification }) {
  return <Badge className={CLASSIFICATION_STYLES[classification]}>{classification}</Badge>;
}

export function ScoreMeter({ score }: { score: number }) {
  const tone = score >= 70 ? "bg-emerald-400" : score >= 40 ? "bg-amber-400" : "bg-subtle";
  return (
    <div className="flex items-center gap-3">
      <div className="h-2 w-32 overflow-hidden rounded-full bg-border">
        <div className={`h-full ${tone}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-sm text-fg">{score}/100</span>
    </div>
  );
}
