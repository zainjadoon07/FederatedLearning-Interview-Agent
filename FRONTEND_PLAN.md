# Frontend Development Plan
## Federated AI Interview Platform

> **Track progress by checking off tasks as you complete them.**
> Backend is fully implemented. This document covers the entire frontend build.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Next.js 14 (App Router) | Framework |
| Tailwind CSS | Styling |
| Framer Motion | Animations & transitions |
| Zustand | Global state management |
| Axios | API calls |
| React Hook Form | Form handling |
| Recharts | Score charts in reports |

---

## Route Structure

```
app/
├── page.tsx                                  ← Root redirect / landing
├── (auth)/
│   ├── login/page.tsx
│   └── register/page.tsx
├── (recruiter)/
│   ├── layout.tsx                            ← Sidebar + auth guard
│   ├── dashboard/page.tsx
│   ├── templates/page.tsx
│   ├── candidate/[session_id]/page.tsx
│   └── settings/page.tsx
└── interview/
    └── [interview_id]/
        ├── page.tsx                          ← Candidate onboarding
        └── live/page.tsx                     ← Live interview loop
```

---

## Pages

---

### 1. `/` — Root Page
- [ ] Check for JWT in localStorage
- [ ] If authenticated → redirect to `/dashboard`
- [ ] If not → show landing hero with Login + Register CTAs
- **API:** None

---

### 2. `/login` — Recruiter Login
- [ ] Email + Password form
- [ ] Inline error handling
- [ ] On success → save JWT to localStorage + Zustand, redirect to `/dashboard`
- [ ] Link to `/register`
- **API:** `POST /api/auth/login`

---

### 3. `/register` — Recruiter Registration
- [ ] Company Name + Email + Password form
- [ ] On success → save JWT, redirect to `/dashboard`
- [ ] Link to `/login`
- **API:** `POST /api/auth/register`

---

### 4. `/dashboard` — Recruiter Dashboard *(Protected)*
- [ ] Stats bar: total candidates, passed, failed, average score
- [ ] Candidates table: Name, Email, Score %, Status badge, Date
- [ ] Status badges: Passed (green) / Failed (red) / Analyzing (yellow) / In Progress (blue)
- [ ] Row click → navigate to `/candidate/[session_id]`
- [ ] Empty state UI when no candidates exist
- [ ] "Create Interview" button → `/templates`
- **API:** `GET /api/reports/candidates`

---

### 5. `/templates` — Interview Template Creator *(Protected)*
- [ ] **Create form:**
  - [ ] Job Role (text input)
  - [ ] Skills Required (tag input — type + Enter to add/remove)
  - [ ] Difficulty (dropdown: Easy / Medium / Hard)
  - [ ] Number of Questions (number input, 1–10)
  - [ ] "Generate Link" button
  - [ ] On success → show shareable link in modal with copy button
- [ ] **Existing templates list:**
  - [ ] Table/cards: Role, Skills, Difficulty, Questions, Date, Copy Link icon
- **API:**
  - `POST /api/templates/create`
  - `GET /api/templates/list`

---

### 6. `/candidate/[session_id]` — Candidate Detail Report *(Protected)*
- [ ] Header: candidate name, email, date, Pass/Fail badge
- [ ] Large final score percentage display
- [ ] Dual-model score chart (DistilBERT vs Gemini per question — bar chart)
- [ ] Q&A breakdown table: Question | Answer | DistilBERT Score | Gemini Score | Final Score
- [ ] Color-coded scores: 0 = red, 1 = yellow, 2 = green
- [ ] Cheating flags section: tab switches, copy-paste attempts, time taken
- [ ] "Back to Dashboard" button
- **API:** `GET /api/reports/candidate/{session_id}`

---

### 7. `/settings` — Company Settings *(Protected)*
- [ ] Passing Threshold slider (0–100, default 70)
- [ ] AI Strictness dropdown (Lenient / Strict / Hyper-Strict)
- [ ] Theme Color picker (hex input + color swatch preview)
- [ ] Company Site URL (text input)
- [ ] Save button with success/error feedback
- [ ] Load current values on page mount
- **API:**
  - `GET /api/settings/get`
  - `PUT /api/settings/update`

---

### 8. `/interview/[interview_id]` — Candidate Onboarding *(Public)*
- [ ] Welcome message with role name (fetch template info or display generic)
- [ ] Form: Full Name, Email, Resume upload (PDF, optional)
- [ ] Duplicate session guard: if email already used → show "already completed" message
- [ ] "Start Interview" button
- [ ] On success → store `session_id` + `all_questions[]` in Zustand, navigate to `/live`
- **API:** `POST /api/sessions/start` *(multipart/form-data)*

---

### 9. `/interview/[interview_id]/live` — Live Interview Loop *(Public)*
- [ ] Progress bar: "Question X of Y"
- [ ] Countdown timer per question
- [ ] Question card displaying current question text
- [ ] Answer textarea (editable)
- [ ] Microphone button (Web Speech API → real-time transcription into textarea)
- [ ] "Next" button (advances question) / "Submit Interview" on last question
- [ ] **Anti-cheat (invisible):**
  - [ ] `window.addEventListener('blur')` → increment `tab_switches`
  - [ ] `onPaste` on textarea → increment `copy_paste_attempts`
  - [ ] Track `time_taken_seconds` from start
- [ ] Zero API calls during interview (all questions from Zustand)
- [ ] On final submit → send answers + cheating flags → show Thank You screen
- **API:** `POST /api/sessions/submit_all`

---

### 10. Thank You Screen *(inline state on `/live`)*
- [ ] Animated spinner / success illustration
- [ ] "Thank you, [Name]! Your interview has been submitted."
- [ ] "Results are being processed. You may close this window."
- **API:** None (evaluation is async on backend)

---

## Shared Components

| Component | Used In | Status |
|---|---|---|
| `Sidebar.tsx` | All recruiter pages | ⬜ Not started |
| `AuthGuard.tsx` | Recruiter layout | ⬜ Not started |
| `Navbar.tsx` | Candidate-facing pages | ⬜ Not started |
| `ScoreBadge.tsx` | Dashboard, report page | ⬜ Not started |
| `DualScoreChart.tsx` | Candidate report | ⬜ Not started |
| `TagInput.tsx` | Template creator | ⬜ Not started |
| `CopyLinkModal.tsx` | Template creator | ⬜ Not started |
| `CountdownTimer.tsx` | Live interview | ⬜ Not started |
| `MicrophoneButton.tsx` | Live interview | ⬜ Not started |
| `AntiCheatTracker.tsx` | Live interview | ⬜ Not started |

---

## Zustand Stores

### `authStore.ts`
- [ ] `token: string | null`
- [ ] `isAuthenticated: boolean`
- [ ] `login(token: string)`
- [ ] `logout()`

### `interviewStore.ts`
- [ ] `sessionId: string | null`
- [ ] `candidateName: string`
- [ ] `questions: string[]`
- [ ] `answers: string[]`
- [ ] `currentIndex: number`
- [ ] `cheatingFlags: { tab_switches, copy_paste_attempts, time_taken_seconds }`
- [ ] `setSession(sessionId, questions, name)`
- [ ] `saveAnswer(index, answer)`
- [ ] `nextQuestion()`
- [ ] `incrementTabSwitch()`
- [ ] `incrementPaste()`

---

## API Reference Summary

| Method | Endpoint | Auth | Used In |
|---|---|---|---|
| POST | `/api/auth/register` | ❌ | Register page |
| POST | `/api/auth/login` | ❌ | Login page |
| PUT | `/api/auth/profile` | ✅ | Settings (optional) |
| DELETE | `/api/auth/delete` | ✅ | Settings (optional) |
| GET | `/api/settings/get` | ✅ | Settings page |
| PUT | `/api/settings/update` | ✅ | Settings page |
| POST | `/api/templates/create` | ✅ | Templates page |
| GET | `/api/templates/list` | ✅ | Templates page |
| POST | `/api/sessions/start` | ❌ | Candidate onboarding |
| POST | `/api/sessions/submit_all` | ❌ | Live interview |
| GET | `/api/reports/candidates` | ✅ | Dashboard |
| GET | `/api/reports/candidate/{session_id}` | ✅ | Candidate report |

---

## Build Order

Work through these phases in order. Each phase is independently testable.

### Phase 1 — Auth Foundation
- [ ] Set up Next.js project with Tailwind + Framer Motion
- [ ] Install Zustand, Axios, React Hook Form, Recharts
- [ ] Create `authStore.ts`
- [ ] Build `/login` page
- [ ] Build `/register` page
- [ ] Root `/` redirect logic

### Phase 2 — Recruiter Shell
- [ ] Build recruiter `layout.tsx` with sidebar
- [ ] Build `AuthGuard.tsx` (redirect to login if no token)
- [ ] Build `Sidebar.tsx` with nav links

### Phase 3 — Recruiter Pages
- [ ] Build `/dashboard` page
- [ ] Build `/templates` page + `TagInput` + `CopyLinkModal`
- [ ] Build `/settings` page
- [ ] Build `/candidate/[session_id]` report page + `DualScoreChart`

### Phase 4 — Candidate Flow
- [ ] Create `interviewStore.ts`
- [ ] Build `/interview/[interview_id]` onboarding page
- [ ] Build `/interview/[interview_id]/live` interview loop
- [ ] Add `CountdownTimer`, `MicrophoneButton`, `AntiCheatTracker`
- [ ] Build Thank You screen state

### Phase 5 — Polish
- [ ] Dark mode / glassmorphism styling
- [ ] Framer Motion page transitions
- [ ] Loading skeletons for data-fetching pages
- [ ] Mobile responsive layout
- [ ] Error boundary components

---

## Notes

- Backend runs on `http://localhost:8000` — update `baseURL` in `src/lib/api.ts`
- All protected routes send `Authorization: Bearer <token>` header (already handled by Axios interceptor in `api.ts`)
- The live interview page must guard against page refresh — store questions in Zustand with `persist` middleware or `sessionStorage`
- Resume upload must use `multipart/form-data`, not JSON — use `FormData` object with Axios
- The backend returns `all_questions[]` upfront at session start — no per-question API calls needed during the interview
- Evaluation is async — the Thank You screen never needs to poll for results
