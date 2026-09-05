import { NewRunForm } from "@/components/NewRunForm";

export default function NewRunPage() {
  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-fg">Start a research run</h1>
        <p className="text-sm text-muted">
          Describe what you want to find — the planner derives search keywords and a date range
          from your objective&apos;s own wording.
        </p>
      </div>
      <div className="card">
        <NewRunForm />
      </div>
    </div>
  );
}
