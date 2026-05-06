// ─── Auth ────────────────────────────────────────────────────────────────────

export interface LoginPayload {
  email: string;
  password: string;
}

export interface RegisterPayload {
  name: string;
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// ─── Settings ────────────────────────────────────────────────────────────────

export interface CompanySettings {
  passing_threshold: number;
  ai_strictness: "lenient" | "strict" | "hyper-strict";
  theme_color: string;
  company_site_url: string | null;
}

// ─── Templates ───────────────────────────────────────────────────────────────

export interface TemplateCreatePayload {
  role: string;
  skills_required: string[];
  difficulty: "easy" | "medium" | "hard";
  total_questions: number;
}

export interface InterviewTemplate {
  interview_id: string;
  company_id: string;
  role: string;
  skills_required: string[];
  difficulty: string;
  total_questions: number;
  shareable_link: string;
  created_at: string;
}

// ─── Sessions ────────────────────────────────────────────────────────────────

export interface SessionStartResponse {
  session_id: string;
  all_questions: string[];
}

export interface CheatingFlags {
  tab_switches: number;
  copy_paste_attempts: number;
  time_taken_seconds: number;
}

export interface BatchSubmitPayload {
  session_id: string;
  answers: string[];
  cheating_flags: CheatingFlags;
}

// ─── Reports ─────────────────────────────────────────────────────────────────

export type SessionStatus = "in_progress" | "analyzing" | "completed";

export interface CandidateSummary {
  session_id: string;
  interview_id: string;
  candidate_name: string;
  candidate_email: string;
  status: SessionStatus;
  final_grade_percentage: number | null;
  is_passed: boolean | null;
  created_at: string;
}

export interface EvaluationEntry {
  question: string;
  answer: string;
  final_score: number;
  model_score: number;
  gemini_score: number | null;
}

export interface CandidateReport extends CandidateSummary {
  questions: string[];
  answers: string[];
  evaluation_history: EvaluationEntry[];
  cheating_flags: CheatingFlags | null;
}
