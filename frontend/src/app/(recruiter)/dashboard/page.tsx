"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { getCandidatesApi } from "@/lib/api";
import type { CandidateSummary, SessionStatus } from "@/types";

// ─── Status Badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status, isPassed }: { status: SessionStatus; isPassed: boolean | null }) {
  if (status === "in_progress") {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/15 text-blue-400 border border-blue-500/25">
        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
        In Progress
      </span>
    );
  }
  if (status === "analyzing") {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-yellow-500/15 text-yellow-400 border border-yellow-500/25">
        <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />
        Analyzing
      </span>
    );
  }
  if (isPassed === true) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
        Passed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/15 text-red-400 border border-red-500/25">
      <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
      Failed
    </span>
  );
}

// ─── Stat Card ────────────────────────────────────────────────────────────────
function StatCard({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className="bg-[#16161f] border border-white/8 rounded-xl p-5">
      <p className="text-zinc-400 text-sm">{label}</p>
      <p className={`text-3xl font-bold mt-1 ${color}`}>{value}</p>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const router = useRouter();
  const [candidates, setCandidates] = useState<CandidateSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCandidatesApi()
      .then(setCandidates)
      .catch(() => setError("Failed to load candidates."))
      .finally(() => setLoading(false));
  }, []);

  const completed = candidates.filter((c) => c.status === "completed");
  const passed = completed.filter((c) => c.is_passed).length;
  const failed = completed.filter((c) => !c.is_passed).length;
  const avgScore =
    completed.length > 0
      ? Math.round(
          completed.reduce((sum, c) => sum + (c.final_grade_percentage ?? 0), 0) /
            completed.length
        )
      : 0;

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-zinc-400 text-sm mt-1">
            All candidates who have taken your interviews
          </p>
        </div>
        <button
          onClick={() => router.push("/templates")}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold px-4 py-2.5 rounded-lg transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Create Interview
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Candidates" value={candidates.length} color="text-white" />
        <StatCard label="Passed" value={passed} color="text-emerald-400" />
        <StatCard label="Failed" value={failed} color="text-red-400" />
        <StatCard label="Avg Score" value={`${avgScore}%`} color="text-indigo-400" />
      </div>

      {/* Table */}
      <div className="bg-[#16161f] border border-white/8 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/8">
          <h2 className="text-white font-semibold">Candidates</h2>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center py-20">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        ) : candidates.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3">
            <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center">
              <svg className="w-6 h-6 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <p className="text-zinc-400 text-sm">No candidates yet</p>
            <button
              onClick={() => router.push("/templates")}
              className="text-indigo-400 hover:text-indigo-300 text-sm font-medium transition-colors"
            >
              Create your first interview →
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/8">
                  <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-6 py-3">
                    Candidate
                  </th>
                  <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-6 py-3">
                    Score
                  </th>
                  <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-6 py-3">
                    Status
                  </th>
                  <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-6 py-3">
                    Date
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {candidates.map((c, i) => (
                  <motion.tr
                    key={c.session_id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.04 }}
                    onClick={() => router.push(`/candidate/${c.session_id}`)}
                    className="hover:bg-white/3 cursor-pointer transition-colors"
                  >
                    <td className="px-6 py-4">
                      <p className="text-white text-sm font-medium">{c.candidate_name}</p>
                      <p className="text-zinc-500 text-xs mt-0.5">{c.candidate_email}</p>
                    </td>
                    <td className="px-6 py-4">
                      {c.final_grade_percentage != null ? (
                        <span className="text-white text-sm font-semibold">
                          {c.final_grade_percentage.toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-zinc-600 text-sm">—</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <StatusBadge status={c.status} isPassed={c.is_passed} />
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-zinc-400 text-sm">
                        {new Date(c.created_at).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
