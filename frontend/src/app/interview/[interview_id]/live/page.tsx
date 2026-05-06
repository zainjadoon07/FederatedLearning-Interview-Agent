"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useInterviewStore } from "@/store/interviewStore";
import { submitAnswersApi } from "@/lib/api";

// ─── Countdown Timer ──────────────────────────────────────────────────────────
function CountdownTimer({
  seconds,
  onExpire,
}: {
  seconds: number;
  onExpire: () => void;
}) {
  const [remaining, setRemaining] = useState(seconds);

  useEffect(() => {
    setRemaining(seconds);
  }, [seconds]);

  useEffect(() => {
    if (remaining <= 0) {
      onExpire();
      return;
    }
    const id = setInterval(() => setRemaining((r) => r - 1), 1000);
    return () => clearInterval(id);
  }, [remaining, onExpire]);

  const mins = Math.floor(remaining / 60);
  const secs = remaining % 60;
  const isUrgent = remaining <= 30;

  return (
    <div
      className={`flex items-center gap-1.5 text-sm font-mono font-semibold px-3 py-1.5 rounded-lg border transition-colors ${
        isUrgent
          ? "text-red-400 bg-red-500/10 border-red-500/25 animate-pulse"
          : "text-zinc-300 bg-white/5 border-white/10"
      }`}
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      {String(mins).padStart(2, "0")}:{String(secs).padStart(2, "0")}
    </div>
  );
}

// ─── Microphone Button ────────────────────────────────────────────────────────
function MicrophoneButton({ onTranscript }: { onTranscript: (text: string) => void }) {
  const [listening, setListening] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);

  const toggle = () => {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SR: any =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SR) {
      alert("Speech recognition is not supported in your browser. Try Chrome.");
      return;
    }

    const recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      onTranscript(transcript);
    };

    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  };

  return (
    <button
      type="button"
      onClick={toggle}
      title={listening ? "Stop recording" : "Start voice input"}
      className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium border transition-all ${
        listening
          ? "bg-red-500/15 text-red-400 border-red-500/30 animate-pulse"
          : "bg-white/5 text-zinc-400 border-white/10 hover:text-white hover:bg-white/10"
      }`}
    >
      <svg className="w-4 h-4" fill={listening ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
      {listening ? "Stop" : "Speak"}
    </button>
  );
}

// ─── Thank You Screen ─────────────────────────────────────────────────────────
function ThankYouScreen({ name }: { name: string }) {
  return (
    <div className="min-h-screen bg-[#0f0f13] flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="text-center max-w-md"
      >
        <div className="w-20 h-20 rounded-full bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center mx-auto mb-6">
          <svg className="w-10 h-10 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h1 className="text-3xl font-bold text-white mb-3">
          Thank you, {name}!
        </h1>
        <p className="text-zinc-400 text-sm leading-relaxed mb-2">
          Your interview has been submitted successfully.
        </p>
        <p className="text-zinc-500 text-sm">
          Results are being processed by our AI. You may close this window.
        </p>
        <div className="mt-8 flex items-center justify-center gap-2 text-zinc-600 text-xs">
          <div className="w-4 h-4 border-2 border-zinc-600 border-t-zinc-400 rounded-full animate-spin" />
          Processing in background...
        </div>
      </motion.div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
const SECONDS_PER_QUESTION = 180; // 3 minutes per question

export default function LiveInterviewPage() {
  const { interview_id } = useParams<{ interview_id: string }>();
  const router = useRouter();

  const {
    sessionId,
    candidateName,
    questions,
    answers,
    currentIndex,
    cheatingFlags,
    saveAnswer,
    nextQuestion,
    incrementTabSwitch,
    incrementPaste,
    setTimeTaken,
    reset,
  } = useInterviewStore();

  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timerKey, setTimerKey] = useState(0); // reset timer on question change
  const startTimeRef = useRef<number>(Date.now());

  // Guard: if no session, redirect back to onboarding
  useEffect(() => {
    if (!sessionId || questions.length === 0) {
      router.replace(`/interview/${interview_id}`);
    }
  }, [sessionId, questions, interview_id, router]);

  // Anti-cheat: tab switch detection
  useEffect(() => {
    const handleBlur = () => incrementTabSwitch();
    window.addEventListener("blur", handleBlur);
    return () => window.removeEventListener("blur", handleBlur);
  }, [incrementTabSwitch]);

  const handleTimerExpire = useCallback(() => {
    // Auto-advance on timer expiry
    if (currentIndex < questions.length - 1) {
      nextQuestion();
      setTimerKey((k) => k + 1);
    }
  }, [currentIndex, questions.length, nextQuestion]);

  const handleNext = () => {
    nextQuestion();
    setTimerKey((k) => k + 1);
  };

  const handleSubmit = async () => {
    if (!sessionId) return;
    setSubmitting(true);
    setError(null);

    const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
    setTimeTaken(elapsed);

    try {
      await submitAnswersApi({
        session_id: sessionId,
        answers,
        cheating_flags: { ...cheatingFlags, time_taken_seconds: elapsed },
      });
      reset(); // clear store after successful submit
      setSubmitted(true);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? "Submission failed. Please try again.";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return <ThankYouScreen name={candidateName} />;
  }

  if (!sessionId || questions.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0f0f13]">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const isLast = currentIndex === questions.length - 1;
  const currentAnswer = answers[currentIndex] ?? "";
  const progress = ((currentIndex + 1) / questions.length) * 100;

  return (
    <div className="min-h-screen bg-[#0f0f13] flex flex-col">
      {/* Top Bar */}
      <div className="border-b border-white/8 bg-[#13131a] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
            <span className="text-white text-xs font-bold">AI</span>
          </div>
          <span className="text-white text-sm font-medium">InterviewAI</span>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-zinc-400 text-sm">
            Question{" "}
            <span className="text-white font-semibold">{currentIndex + 1}</span>
            {" "}of{" "}
            <span className="text-white font-semibold">{questions.length}</span>
          </span>
          <CountdownTimer
            key={timerKey}
            seconds={SECONDS_PER_QUESTION}
            onExpire={handleTimerExpire}
          />
        </div>
      </div>

      {/* Progress Bar */}
      <div className="h-1 bg-white/5">
        <motion.div
          className="h-full bg-indigo-500"
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.4 }}
        />
      </div>

      {/* Content */}
      <div className="flex-1 flex items-center justify-center px-4 py-10">
        <div className="w-full max-w-2xl">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentIndex}
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -30 }}
              transition={{ duration: 0.3 }}
            >
              {/* Question Card */}
              <div className="bg-[#16161f] border border-white/8 rounded-2xl p-6 mb-5">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xs font-medium text-indigo-400 bg-indigo-600/15 border border-indigo-500/25 px-2.5 py-1 rounded-full">
                    Question {currentIndex + 1}
                  </span>
                </div>
                <p className="text-white text-lg leading-relaxed font-medium">
                  {questions[currentIndex]}
                </p>
              </div>

              {/* Answer Area */}
              <div className="bg-[#16161f] border border-white/8 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-3">
                  <label className="text-sm font-medium text-zinc-300">
                    Your Answer
                  </label>
                  <MicrophoneButton
                    onTranscript={(text) =>
                      saveAnswer(currentIndex, currentAnswer + text)
                    }
                  />
                </div>

                <textarea
                  value={currentAnswer}
                  onChange={(e) => saveAnswer(currentIndex, e.target.value)}
                  onPaste={() => incrementPaste()}
                  placeholder="Type your answer here, or use the microphone button to speak..."
                  rows={6}
                  className="w-full bg-[#1e1e2e] border border-white/10 rounded-lg px-4 py-3 text-white placeholder-zinc-600 text-sm outline-none transition resize-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                />

                {error && (
                  <div className="mt-3 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
                    <p className="text-red-400 text-sm">{error}</p>
                  </div>
                )}

                <div className="flex items-center justify-between mt-4">
                  <p className="text-zinc-600 text-xs">
                    {currentAnswer.length} characters
                  </p>

                  {isLast ? (
                    <button
                      onClick={handleSubmit}
                      disabled={submitting}
                      className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold px-6 py-2.5 rounded-lg transition-colors text-sm"
                    >
                      {submitting ? (
                        <>
                          <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Submitting...
                        </>
                      ) : (
                        <>
                          Submit Interview
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        </>
                      )}
                    </button>
                  ) : (
                    <button
                      onClick={handleNext}
                      className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-2.5 rounded-lg transition-colors text-sm"
                    >
                      Next Question
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
