"use client";

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { motion } from "framer-motion";
import { getSettingsApi, updateSettingsApi } from "@/lib/api";
import type { CompanySettings } from "@/types";

export default function SettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit, watch, setValue, reset } =
    useForm<CompanySettings>({
      defaultValues: {
        passing_threshold: 70,
        ai_strictness: "strict",
        theme_color: "#2563EB",
        company_site_url: "",
      },
    });

  const threshold = watch("passing_threshold");
  const themeColor = watch("theme_color");

  useEffect(() => {
    getSettingsApi()
      .then((data) => reset({ ...data, company_site_url: data.company_site_url ?? "" }))
      .catch(() => setError("Failed to load settings."))
      .finally(() => setLoading(false));
  }, [reset]);

  const onSubmit = async (data: CompanySettings) => {
    setSaving(true);
    setError(null);
    setSuccess(false);
    try {
      await updateSettingsApi({
        ...data,
        passing_threshold: Number(data.passing_threshold),
        company_site_url: data.company_site_url || null,
      });
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch {
      setError("Failed to save settings.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-8 max-w-2xl">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Configure AI behavior and company preferences
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Passing Threshold */}
        <div className="bg-[#16161f] border border-white/8 rounded-xl p-6">
          <h2 className="text-white font-semibold mb-4">Evaluation Rules</h2>

          <div className="space-y-5">
            {/* Threshold Slider */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-zinc-300">
                  Passing Threshold
                </label>
                <span className="text-indigo-400 font-bold text-sm">{threshold}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                className="w-full accent-indigo-500 cursor-pointer"
                {...register("passing_threshold")}
              />
              <div className="flex justify-between text-xs text-zinc-600 mt-1">
                <span>0%</span>
                <span>Candidates must score at least {threshold}% to pass</span>
                <span>100%</span>
              </div>
            </div>

            {/* AI Strictness */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                AI Strictness
              </label>
              <select
                className="w-full bg-[#1e1e2e] border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                {...register("ai_strictness")}
              >
                <option value="lenient">Lenient — Rewards partial attempts</option>
                <option value="strict">Strict — Balanced and unbiased</option>
                <option value="hyper-strict">Hyper-Strict — Extremely harsh</option>
              </select>
              <p className="text-zinc-500 text-xs mt-1.5">
                This setting cascades into the Gemini evaluation prompt for every interview.
              </p>
            </div>
          </div>
        </div>

        {/* Branding */}
        <div className="bg-[#16161f] border border-white/8 rounded-xl p-6">
          <h2 className="text-white font-semibold mb-4">Branding</h2>

          <div className="space-y-5">
            {/* Theme Color */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Theme Color
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="color"
                  className="w-10 h-10 rounded-lg border border-white/10 bg-transparent cursor-pointer"
                  {...register("theme_color")}
                />
                <input
                  type="text"
                  value={themeColor}
                  onChange={(e) => setValue("theme_color", e.target.value)}
                  className="flex-1 bg-[#1e1e2e] border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 font-mono"
                  placeholder="#2563EB"
                />
                <div
                  className="w-10 h-10 rounded-lg border border-white/10 flex-shrink-0"
                  style={{ backgroundColor: themeColor }}
                />
              </div>
            </div>

            {/* Company Site URL */}
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                Company Site URL
              </label>
              <input
                type="url"
                placeholder="https://yourcompany.com"
                className="w-full bg-[#1e1e2e] border border-white/10 rounded-lg px-4 py-2.5 text-white placeholder-zinc-500 text-sm outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                {...register("company_site_url")}
              />
            </div>
          </div>
        </div>

        {/* Feedback */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
        {success && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-4 py-3"
          >
            <p className="text-emerald-400 text-sm">Settings saved successfully.</p>
          </motion.div>
        )}

        <button
          type="submit"
          disabled={saving}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold px-6 py-2.5 rounded-lg transition-colors text-sm"
        >
          {saving && (
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          )}
          {saving ? "Saving..." : "Save Settings"}
        </button>
      </form>
    </div>
  );
}
