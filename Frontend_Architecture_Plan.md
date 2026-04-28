# 🖥️ Federated AI Interview Platform: Frontend Architecture Plan

This document serves as the master blueprint for the web-engineering phase of the project. It outlines the core features, UI/UX guidelines, and exact mappings to our backend API endpoints.

---

## 🏗️ 1. Core Technology Stack
- **Framework:** Next.js (React) using the App Router for optimal SEO and performance.
- **Styling:** Tailwind CSS + Framer Motion (For extremely smooth, glassmorphic UI animations and page transitions).
- **State Management:** React Context or Zustand (to manage the interview loop state locally without re-rendering the whole app).
- **Forms & Fetching:** React Hook Form + Axios (or native fetch).

---

## 💼 2. Module A: The Recruiter Dashboard (Protected Hub)

This side of the platform is strictly for companies. It requires JWT token management.

### A. Authentication Portal
- **Features:** A sleek, split-screen login/registration page with smooth error handling.
- **API Mapping:** 
  - `POST /api/auth/login`
  - `POST /api/auth/register`
- **UX Goal:** Must feel like an enterprise SaaS (e.g., Vercel or Stripe).

### B. Global Overview Dashboard
- **Features:** A beautiful data table showing all applicants. Includes visual badges (Green "Passed", Red "Failed") and a high-level summary chart.
- **API Mapping:** `GET /api/reports/candidates`

### C. The Detail View (Candidate Report)
- **Features:** Clicking a candidate in the dashboard opens a massive report. It will render the "Score Difference Chart" comparing DistilBERT to Gemini, and show exactly what the candidate typed versus what cheating flags were triggered.
- **API Mapping:** `GET /api/reports/candidate/{session_id}`

### D. Interview Blueprint Creator
- **Features:** A modern form where HR can type the Job Title, add skills like tags, and hit "Generate Link". 
- **API Mapping:** `POST /api/templates/create`

### E. Company Settings Panel
- **Features:** Sliders to adjust "Passing Threshold", and dropdowns to adjust "AI Strictness". 
- **API Mapping:** `GET /api/settings/get` and `PUT /api/settings/update`

---

## 🧑‍💻 3. Module B: The Candidate Interview Engine (Public Hub)

This is the Shareable Link flow. It must be extremely robust, preventing data loss if the browser refreshes.

### A. The Landing / Onboarding Page
- **Routing:** `/interview/[interview_id]` (Dynamic Next.js Route).
- **Features:** A welcoming page asking for Name, Email, and a file upload block for their PDF Resume.
- **API Mapping:** `POST /api/sessions/start` (Must be sent as `multipart/form-data`).

### B. The Live Interview Loop (The Magic UI)
- **Features:** 
  - Since the backend returned an array of `all_questions`, the React app maps over this array locally.
  - Features a clean progress bar at the top (e.g., "Question 2 of 5").
  - Includes a **Countdown Timer** to pressure the candidate visually.
- **Unique FYP Feature 1: Native Speech-to-Text**
  - A microphone button that utilizes the native browser `Web Speech API`. 
  - The candidate speaks, the text types itself out in real-time in the text box, and they can edit it manually before hitting "Next".
- **Unique FYP Feature 2: React Anti-Cheat Hooks**
  - The frontend runs `window.addEventListener('blur')` to count how many times the user switched to ChatGPT. 
  - It intercepts `onPaste` events inside the textarea to track if they pasted pre-written code.
- **API Mapping:** NONE! Zero latency because the UI is completely isolated.

### C. The Final Submission / Thank You Screen
- **Features:** Once the final question is answered, the app gathers all the typed strings into one big array, appends the cheating analytics, and fires the submission payload. It displays a spinning "Thank You! Processing Results..." screen.
- **API Mapping:** `POST /api/sessions/submit_all`

---

## 🎨 4. General Aesthetic Guidelines
To score top marks on presentation:
- **No boring white screens:** Use sleek dark modes, subtle glowing gradients, and glassmorphism.
- **Typography:** Use modern fonts like `Inter` or `Outfit`.
- **Micro-interactions:** Buttons should scale slightly when hovered; pages should slide or fade into each other using Framer Motion. 
