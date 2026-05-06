"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getCandidateReportApi } from "@/lib/api";
import type { CandidateReport } from "@/types";

// ─── Score label helper ───────────────────────────────────────────────────────
const scoreLabel = (s: number) => ["Poor", "Average", "Excellent"][s] ?? "—";
const scoreColor = (s: number) =>
  ["text-red-400", "text-yellow-400", "text-emerald-400"][s] ?? "text-zinc-400";
const scoreBg = (s: number) =>
  ["bg-red-500/10 border-red-500/25", "bg-yellow-500/10 border-yellow-500/25", "bg-emerald-500/10 border-emerald-500/25"][s] ?? "";

export default function CandidateReportPage() {
  const { session_id } = useParams<{ session_id: string }>();
  const router = useRouter();
  const [report, setReport] = useState<CandidateReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCandidateReportApi(session_id)
      .then(setReport)
      .catch(() => setError("Failed to load report."))
      .finally(() => setLoading(false));
  }, [session_id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-3">
        <p className="text-red-400">{error ?? "Report not found."}</p>
        <button onClick={() => router.back()} className="text-indigo-400 text-sm hover:underline">
          ← Go back
        </button>
      </div>
    );
  }

  // Build chart data
  const chartData = (report.evaluation_history ?? []).map((entry, i) => ({
    name: `Q${i + 1}`,
    DistilBERT: entry.model_score,
    Gemini: entry.gemini_score ?? entry.model_score,
    Final: entry.final_score,
  }));

  const isPassed = report.is_passed;
  const score = report.final_grade_percentage;
  const isAnalyzing = report.status === "analyzing";

  return (
    <div className="p-8 max-w-5xl">
      {/* Back */}
      <button
        onClick={() => router.push("/dashboard")}
        className="flex items-center gap-1.5 text-zinc-400 hover:text-white text-sm mb-6 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Dashboard
      </button>

      {/* Header Card */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-[#16161f] border border-white/8 rounded-xl p-6 mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <div>
          <h1 className="text-2xl font-bold text-white">{report.candidate_name}</h1>
          <p className="text-zinc-400 text-sm mt-0.5">{report.candidate_email}</p>
          <p className="text-zinc-500 text-xs mt-1">
            {new Date(report.created_at).toLocaleDateString("en-US", {
              weekday: "long", year: "numeric", month: "long", day: "numeric",
            })}
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* Score */}
          <div className="text-center">
            <p className="text-4xl font-bold text-white">
              {isAnalyzing ? "—" : `${score?.toFixed(1)}%`}
            </p>
            <p className="text-zinc-500 text-xs mt-0.5">Final Score</p>
          </div>

          {/* Pass/Fail */}
          {isAnalyzing ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium bg-yellow-500/15 text-yellow-400 border border-yellow-500/25">
              <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
              Analyzing
            </span>
          ) : isPassed ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
              ✓ Passed
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium bg-red-500/15 text-red-400 border border-red-500/25">
              ✗ Failed
            </span>
          )}
        </div>
      </motion.div>

      {isAnalyzing ? (
        <div className="bg-[#16161f] border border-white/8 rounded-xl p-12 flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-zinc-400 text-sm">AI evaluation is still running...</p>
          <p className="text-zinc-600 text-xs">Refresh the page in a moment</p>
        </div>
      ) : (
        <>
          {/* Dual Score Chart */}
          {chartData.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-[#16161f] border border-white/8 rounded-xl p-6 mb-6"
            >
              <h2 className="text-white font-semibold mb-1">Model Score Comparison</h2>
              <p className="text-zinc-500 text-xs mb-5">
                DistilBERT (local) vs Gemini (cloud) score per question — scale 0–2
              </p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff0f" />
                  <XAxis dataKey="name" tick={{ fill: "#71717a", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 2]} ticks={[0, 1, 2]} tick={{ fill: "#71717a", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#1e1e2e", border: "1px solid #ffffff15", borderRadius: 8, color: "#fff", fontSize: 12 }}
                    formatter={(value, name) => {
                      const v = typeof value === "number" ? value : 0;
                      return [`${v} (${scoreLabel(v)})`, name];
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }} />
                  <Bar dataKey="DistilBERT" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Gemini" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          )}

          {/* Q&A Breakdown */}
          {report.evaluation_history?.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="bg-[#16161f] border border-white/8 rounded-xl p-6 mb-6"
            >
              <h2 className="text-white font-semibold mb-5">Answer Breakdown</h2>
              <div className="space-y-4">
                {report.evaluation_history.map((entry, i) => (
                  <div key={i} className="bg-[#1e1e2e] border border-white/8 rounded-lg p-4">
                    <div className="flex items-start justify-between gap-3 mb-3">
                      <p className="text-zinc-300 text-sm font-medium flex-1">
                        <span className="text-zinc-500 mr-2">Q{i + 1}.</span>
                        {entry.question}
                      </p>
                      <span className={`flex-shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full border ${scoreBg(entry.final_score)} ${scoreColor(entry.final_score)}`}>
                        {scoreLabel(entry.final_score)}
                      </span>
                    </div>
                    <p className="text-zinc-400 text-sm bg-black/20 rounded-lg px-3 py-2 mb-3">
                      {entry.answer || <span className="text-zinc-600 italic">No answer provided</span>}
                    </p>
                    <div className="flex items-center gap-4 text-xs text-zinc-500">
                      <span>DistilBERT: <span className={scoreColor(entry.model_score)}>{scoreLabel(entry.model_score)}</span></span>
                      <span>·</span>
                      <span>Gemini: <span className={scoreColor(entry.gemini_score ?? entry.model_score)}>{scoreLabel(entry.gemini_score ?? entry.model_score)}</span></span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Cheating Flags */}
          {report.cheating_flags && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-[#16161f] border border-white/8 rounded-xl p-6"
            >
              <h2 className="text-white font-semibold mb-4">Integrity Report</h2>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-[#1e1e2e] border border-white/8 rounded-lg p-4 text-center">
                  <p className={`text-2xl font-bold ${report.cheating_flags.tab_switches > 3 ? "text-red-400" : "text-white"}`}>
                    {report.cheating_flags.tab_switches}
                  </p>
                  <p className="text-zinc-500 text-xs mt-1">Tab Switches</p>
                </div>
                <div className="bg-[#1e1e2e] border border-white/8 rounded-lg p-4 text-center">
                  <p className={`text-2xl font-bold ${report.cheating_flags.copy_paste_attempts > 0 ? "text-yellow-400" : "text-white"}`}>
                    {report.cheating_flags.copy_paste_attempts}
                  </p>
                  <p className="text-zinc-500 text-xs mt-1">Paste Attempts</p>
                </div>
                <div className="bg-[#1e1e2e] border border-white/8 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-white">
                    {Math.floor(report.cheating_flags.time_taken_seconds / 60)}m{" "}
                    {report.cheating_flags.time_taken_seconds % 60}s
                  </p>
                  <p className="text-zinc-500 text-xs mt-1">Time Taken</p>
                </div>
              </div>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}
