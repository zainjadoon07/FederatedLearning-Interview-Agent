"use client";

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { motion, AnimatePresence } from "framer-motion";
import { createTemplateApi, listTemplatesApi } from "@/lib/api";
import type { InterviewTemplate, TemplateCreatePayload } from "@/types";

// ─── Tag Input ────────────────────────────────────────────────────────────────
function TagInput({
  tags,
  onChange,
}: {
  tags: string[];
  onChange: (tags: string[]) => void;
}) {
  const [input, setInput] = useState("");

  const addTag = () => {
    const trimmed = input.trim();
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed]);
    }
    setInput("");
  };

  const removeTag = (tag: string) => onChange(tags.filter((t) => t !== tag));

  return (
    <div className="bg-[#1e1e2e] border border-white/10 rounded-lg px-3 py-2 min-h-[44px] flex flex-wrap gap-2 items-center focus-within:border-indigo-500 focus-within:ring-1 focus-within:ring-indigo-500 transition">
      {tags.map((tag) => (
        <span
          key={tag}
          className="flex items-center gap-1 bg-indigo-600/25 text-indigo-300 text-xs font-medium px-2.5 py-1 rounded-full border border-indigo-500/30"
        >
          {tag}
          <button
            type="button"
            onClick={() => removeTag(tag)}
            className="text-indigo-400 hover:text-white transition-colors ml-0.5"
          >
            ×
          </button>
        </span>
      ))}
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            addTag();
          }
          if (e.key === "Backspace" && !input && tags.length > 0) {
            onChange(tags.slice(0, -1));
          }
        }}
        onBlur={addTag}
        placeholder={tags.length === 0 ? "Type a skill and press Enter..." : ""}
        className="flex-1 min-w-[140px] bg-transparent text-white text-sm placeholder-zinc-500 outline-none"
      />
    </div>
  );
}

// ─── Copy Link Modal ──────────────────────────────────────────────────────────
function CopyLinkModal({
  link,
  role,
  onClose,
}: {
  link: string;
  role: string;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 px-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-[#16161f] border border-white/10 rounded-2xl p-6 w-full max-w-md shadow-2xl"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-emerald-500/15 flex items-center justify-center">
            <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Interview Created!</h3>
            <p className="text-zinc-400 text-sm">Share this link with candidates for <span className="text-white">{role}</span></p>
          </div>
        </div>

        <div className="bg-[#1e1e2e] border border-white/10 rounded-lg px-4 py-3 flex items-center gap-3 mb-4">
          <p className="text-zinc-300 text-sm flex-1 truncate">{link}</p>
          <button
            onClick={copy}
            className="flex-shrink-0 text-xs font-medium px-3 py-1.5 rounded-md bg-indigo-600/20 text-indigo-400 hover:bg-indigo-600/30 border border-indigo-500/30 transition-colors"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>

        <button
          onClick={onClose}
          className="w-full bg-white/5 hover:bg-white/10 text-zinc-300 text-sm font-medium py-2.5 rounded-lg transition-colors"
        >
          Done
        </button>
      </motion.div>
    </div>
  );
}

// ─── Form Types ───────────────────────────────────────────────────────────────
interface FormValues {
  role: string;
  difficulty: "easy" | "medium" | "hard";
  total_questions: number;
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function TemplatesPage() {
  const [skills, setSkills] = useState<string[]>([]);
  const [skillsError, setSkillsError] = useState(false);
  const [templates, setTemplates] = useState<InterviewTemplate[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [createdLink, setCreatedLink] = useState<{ link: string; role: string } | null>(null);

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormValues>({ defaultValues: { difficulty: "medium", total_questions: 5 } });

  useEffect(() => {
    listTemplatesApi()
      .then(setTemplates)
      .finally(() => setLoadingTemplates(false));
  }, []);

  const onSubmit = async (data: FormValues) => {
    if (skills.length === 0) {
      setSkillsError(true);
      return;
    }
    setSkillsError(false);
    setServerError(null);
    setSubmitting(true);

    const payload: TemplateCreatePayload = {
      role: data.role,
      skills_required: skills,
      difficulty: data.difficulty,
      total_questions: Number(data.total_questions),
    };

    try {
      const created = await createTemplateApi(payload);
      setTemplates((prev) => [created, ...prev]);
      setCreatedLink({ link: created.shareable_link, role: created.role });
      reset();
      setSkills([]);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? "Failed to create template.";
      setServerError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  const copyLink = (link: string) => {
    navigator.clipboard.writeText(link);
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Interview Templates</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Create interview blueprints and share links with candidates
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* ── Create Form ── */}
        <div className="bg-[#16161f] border border-white/8 rounded-xl p-6">
          <h2 className="text-white font-semibold mb-5">Create New Template</h2>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            {/* Role */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Job Role
              </label>
              <input
                type="text"
                placeholder="e.g. Senior React Developer"
                className={`w-full bg-[#1e1e2e] border rounded-lg px-4 py-2.5 text-white placeholder-zinc-500 text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 ${
                  errors.role ? "border-red-500" : "border-white/10"
                }`}
                {...register("role", { required: "Job role is required" })}
              />
              {errors.role && (
                <p className="text-red-400 text-xs mt-1">{errors.role.message}</p>
              )}
            </div>

            {/* Skills */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Skills Required
              </label>
              <TagInput tags={skills} onChange={setSkills} />
              {skillsError && (
                <p className="text-red-400 text-xs mt-1">Add at least one skill</p>
              )}
            </div>

            {/* Difficulty + Questions */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                  Difficulty
                </label>
                <select
                  className="w-full bg-[#1e1e2e] border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  {...register("difficulty")}
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                  Questions
                </label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  className={`w-full bg-[#1e1e2e] border rounded-lg px-4 py-2.5 text-white text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 ${
                    errors.total_questions ? "border-red-500" : "border-white/10"
                  }`}
                  {...register("total_questions", {
                    required: true,
                    min: { value: 1, message: "Min 1" },
                    max: { value: 10, message: "Max 10" },
                  })}
                />
                {errors.total_questions && (
                  <p className="text-red-400 text-xs mt-1">
                    {errors.total_questions.message ?? "Required"}
                  </p>
                )}
              </div>
            </div>

            {serverError && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
                <p className="text-red-400 text-sm">{serverError}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-lg transition-colors text-sm flex items-center justify-center gap-2"
            >
              {submitting && (
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              )}
              {submitting ? "Generating..." : "Generate Interview Link"}
            </button>
          </form>
        </div>

        {/* ── Existing Templates ── */}
        <div className="bg-[#16161f] border border-white/8 rounded-xl p-6">
          <h2 className="text-white font-semibold mb-5">Existing Templates</h2>

          {loadingTemplates ? (
            <div className="flex items-center justify-center py-16">
              <div className="w-7 h-7 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : templates.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 gap-2">
              <p className="text-zinc-500 text-sm">No templates yet</p>
              <p className="text-zinc-600 text-xs">Create one using the form</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
              {templates.map((t, i) => (
                <motion.div
                  key={t.interview_id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="bg-[#1e1e2e] border border-white/8 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-white text-sm font-medium truncate">{t.role}</p>
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {t.skills_required.map((s) => (
                          <span
                            key={s}
                            className="text-xs bg-white/5 text-zinc-400 px-2 py-0.5 rounded-full border border-white/8"
                          >
                            {s}
                          </span>
                        ))}
                      </div>
                      <div className="flex items-center gap-3 mt-2">
                        <span className="text-xs text-zinc-500 capitalize">{t.difficulty}</span>
                        <span className="text-zinc-700">·</span>
                        <span className="text-xs text-zinc-500">{t.total_questions} questions</span>
                        <span className="text-zinc-700">·</span>
                        <span className="text-xs text-zinc-500">
                          {new Date(t.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => copyLink(t.shareable_link)}
                      title="Copy link"
                      className="flex-shrink-0 p-2 rounded-lg bg-white/5 hover:bg-indigo-600/20 text-zinc-400 hover:text-indigo-400 border border-white/8 hover:border-indigo-500/30 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Copy Link Modal */}
      <AnimatePresence>
        {createdLink && (
          <CopyLinkModal
            link={createdLink.link}
            role={createdLink.role}
            onClose={() => setCreatedLink(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
