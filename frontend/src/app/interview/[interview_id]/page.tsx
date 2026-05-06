"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { startSessionApi } from "@/lib/api";
import { useInterviewStore } from "@/store/interviewStore";

interface OnboardingForm {
  candidate_name: string;
  candidate_email: string;
  resume?: FileList;
}

export default function CandidateOnboardingPage() {
  const { interview_id } = useParams<{ interview_id: string }>();
  const router = useRouter();
  const { setSession } = useInterviewStore();

  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<OnboardingForm>();

  const onSubmit = async (data: OnboardingForm) => {
    setServerError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("interview_id", interview_id);
    formData.append("candidate_name", data.candidate_name);
    formData.append("candidate_email", data.candidate_email);
    if (data.resume && data.resume.length > 0) {
      formData.append("resume", data.resume[0]);
    }

    try {
      const res = await startSessionApi(formData);
      setSession(res.session_id, res.all_questions, data.candidate_name);
      router.push(`/interview/${interview_id}/live`);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
      if (detail?.toLowerCase().includes("already exists")) {
        setServerError(
          "You have already completed this interview. Each candidate can only attempt once."
        );
      } else {
        setServerError(detail ?? "Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f0f13] flex items-center justify-center px-4 py-12">
      {/* Background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-indigo-600/10 rounded-full blur-3xl" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="relative w-full max-w-lg"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 bg-indigo-600/15 border border-indigo-500/25 rounded-full px-4 py-1.5 mb-4">
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
            <span className="text-indigo-300 text-xs font-medium">AI-Powered Interview</span>
          </div>
          <h1 className="text-3xl font-bold text-white">You&apos;re Invited</h1>
          <p className="text-zinc-400 mt-2 text-sm">
            Fill in your details to begin. The interview is fully automated and takes about 15–20 minutes.
          </p>
        </div>

        {/* Card */}
        <div className="bg-[#16161f] border border-white/10 rounded-2xl p-8 shadow-2xl">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            {/* Name */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Full Name
              </label>
              <input
                type="text"
                placeholder="John Smith"
                className={`w-full bg-[#1e1e2e] border rounded-lg px-4 py-2.5 text-white placeholder-zinc-500 text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 ${
                  errors.candidate_name ? "border-red-500" : "border-white/10"
                }`}
                {...register("candidate_name", { required: "Full name is required" })}
              />
              {errors.candidate_name && (
                <p className="text-red-400 text-xs mt-1">{errors.candidate_name.message}</p>
              )}
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Email Address
              </label>
              <input
                type="email"
                placeholder="you@example.com"
                className={`w-full bg-[#1e1e2e] border rounded-lg px-4 py-2.5 text-white placeholder-zinc-500 text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:5ring-indigo-500 ${
                  errors.candidate_email ? "border-red-500" : "border-white/10"
                }`}
                {...register("candidate_email", {
                  required: "Email is required",
                  pattern: {
                    value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                    message: "Enter a valid email",
                  },
                })}
              />
              {errors.candidate_email && (
                <p className="text-red-400 text-xs mt-1">{errors.candidate_email.message}</p>
              )}
            </div>

            {/* Resume Upload */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Resume{" "}
                <span className="text-zinc-500 font-normal">(optional — PDF only)</span>
              </label>
              <label className="flex items-center gap-3 bg-[#1e1e2e] border border-white/10 border-dashed rounded-lg px-4 py-3 cursor-pointer hover:border-indigo-500/50 transition-colors group">
                <svg className="w-5 h-5 text-zinc-500 group-hover:text-indigo-400 transition-colors flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span className="text-sm text-zinc-400 group-hover:text-zinc-300 transition-colors">
                  {fileName ?? "Click to upload your resume"}
                </span>
                <input
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  {...register("resume")}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    setFileName(file?.name ?? null);
                  }}
                />
              </label>
              <p className="text-zinc-600 text-xs mt-1">
                Uploading your resume allows the AI to ask personalized questions about your experience.
              </p>
            </div>

            {/* Server error */}
            {serverError && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
                <p className="text-red-400 text-sm">{serverError}</p>
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors text-sm flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Preparing your interview...
                </>
              ) : (
                <>
                  Start Interview
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </>
              )}
            </button>
          </form>

          {/* Info */}
          <div className="mt-6 pt-5 border-t border-white/8 grid grid-cols-3 gap-3 text-center">
            {[
              { icon: "🤖", label: "AI-Powered" },
              { icon: "🔒", label: "Secure" },
              { icon: "⚡", label: "15–20 min" },
            ].map((item) => (
              <div key={item.label}>
                <p className="text-lg">{item.icon}</p>
                <p className="text-zinc-500 text-xs mt-0.5">{item.label}</p>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
