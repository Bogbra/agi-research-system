export type RunStatus =
  | "initialization"
  | "planning"
  | "discovery"
  | "evaluation"
  | "completion"
  | "evaluation_failed"
  | "failed";

export type Classification = "Low AGI Potential" | "Medium AGI Potential" | "High AGI Potential";

export interface RunSummary {
  request_id: string;
  research_objective: string;
  status: RunStatus;
  paper_count: number;
  average_agi_score: number | null;
  created_at: string;
  updated_at: string;
}

export interface EvaluatedPaperView {
  title: string;
  link: string;
  authors: string[];
  agi_score: number | null;
  classification: Classification | null;
  overall_assessment: string;
  key_innovations: string[];
}

export interface EvaluationFailureView {
  paper_id: string;
  paper_title: string;
  error_type: string;
  error_message: string;
  attempts: number;
}

export interface RunDetail {
  request_id: string;
  research_objective: string;
  status: RunStatus;
  paper_count: number;
  average_agi_score: number | null;
  final_report: string | null;
  evaluated_papers: EvaluatedPaperView[];
  evaluation_failures: EvaluationFailureView[];
  errors: string[];
  error: string | null;
  execution_plan: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}
