import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CheatingFlags } from "@/types";

interface InterviewState {
  sessionId: string | null;
  candidateName: string;
  questions: string[];
  answers: string[];
  currentIndex: number;
  cheatingFlags: CheatingFlags;

  // Actions
  setSession: (sessionId: string, questions: string[], candidateName: string) => void;
  saveAnswer: (index: number, answer: string) => void;
  nextQuestion: () => void;
  incrementTabSwitch: () => void;
  incrementPaste: () => void;
  setTimeTaken: (seconds: number) => void;
  reset: () => void;
}

const defaultCheatingFlags: CheatingFlags = {
  tab_switches: 0,
  copy_paste_attempts: 0,
  time_taken_seconds: 0,
};

export const useInterviewStore = create<InterviewState>()(
  persist(
    (set, get) => ({
      sessionId: null,
      candidateName: "",
      questions: [],
      answers: [],
      currentIndex: 0,
      cheatingFlags: { ...defaultCheatingFlags },

      setSession: (sessionId, questions, candidateName) =>
        set({
          sessionId,
          questions,
          candidateName,
          answers: new Array(questions.length).fill(""),
          currentIndex: 0,
          cheatingFlags: { ...defaultCheatingFlags },
        }),

      saveAnswer: (index, answer) =>
        set((state) => {
          const updated = [...state.answers];
          updated[index] = answer;
          return { answers: updated };
        }),

      nextQuestion: () =>
        set((state) => ({ currentIndex: state.currentIndex + 1 })),

      incrementTabSwitch: () =>
        set((state) => ({
          cheatingFlags: {
            ...state.cheatingFlags,
            tab_switches: state.cheatingFlags.tab_switches + 1,
          },
        })),

      incrementPaste: () =>
        set((state) => ({
          cheatingFlags: {
            ...state.cheatingFlags,
            copy_paste_attempts: state.cheatingFlags.copy_paste_attempts + 1,
          },
        })),

      setTimeTaken: (seconds) =>
        set((state) => ({
          cheatingFlags: {
            ...state.cheatingFlags,
            time_taken_seconds: seconds,
          },
        })),

      reset: () =>
        set({
          sessionId: null,
          candidateName: "",
          questions: [],
          answers: [],
          currentIndex: 0,
          cheatingFlags: { ...defaultCheatingFlags },
        }),
    }),
    {
      name: "interview-storage",
      storage: {
        getItem: (name) => {
          if (typeof window === "undefined") return null;
          const val = sessionStorage.getItem(name);
          return val ? JSON.parse(val) : null;
        },
        setItem: (name, value) => {
          if (typeof window !== "undefined") {
            sessionStorage.setItem(name, JSON.stringify(value));
          }
        },
        removeItem: (name) => {
          if (typeof window !== "undefined") {
            sessionStorage.removeItem(name);
          }
        },
      },
    }
  )
);
