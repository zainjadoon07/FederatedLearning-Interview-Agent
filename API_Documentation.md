# 🚀 Federated AI Interview Platform: Architecture & API Reference

This document outlines the complete REST API architecture for the Backend system. It details the precise AI inference pipelines, data integration structures, and enterprise routing necessary for the Next.js frontend to securely interface with the system.

---

## 🏢 1. Core System & Security (`/api/auth` & `/api/settings`)
*This module handles multi-tenant isolation, BCrypt password hashing, and global JWT configuration states.*

### `POST /api/auth/register` & `POST /api/auth/login`
- **Internal Logic:** Verifies identity computationally against the MongoDB cluster, validates BCrypt hashes, and generates a stateless stateless `JWT Bearer Access Token` with a configured expiration time.
- **Request Body:** `{ "email": "hr@company.com", "password": "securepass" }`
- **Returns:** `{ "access_token": "eyJhb...", "token_type": "bearer" }`

### `GET /api/settings/get` 🔒 *(Protected)*
- **Internal Logic:** Securely queries the `$company_settings` index using the decoded JWT payload to provide frontend rendering engines with the company's preferred UI colors and AI strictness configurations.
- **Returns:** 
  ```json
  {
    "passing_threshold": 70,
    "ai_strictness": "strict",
    "theme_color": "#2563EB",
    "company_site_url": null
  }
  ```

### `PUT /api/settings/update` 🔒 *(Protected)*
- **Internal Logic:** Handles deep updates. If `ai_strictness` is modified, this endpoint cascades those rules into the physical system prompts governing the Gemini LLM inference pipeline.
- **Request Body:** `{ "passing_threshold": 85, "ai_strictness": "lenient", "theme_color": "#FF0000" }`

---

## 📝 2. Procedural Template Engine (`/api/templates`)
*This module creates the blueprint parameters that the Generative AI uses to structure its questions.*

### `POST /api/templates/create` 🔒 *(Protected)*
- **Internal Logic:** Translates the Recruiter's requirements into an automated blueprint. It binds the created template forcefully to the company's UUID, guaranteeing data isolation.
- **Request Body:** 
  ```json
  { 
    "role": "Next.js Dev", 
    "skills_required": ["React", "TypeScript"], 
    "difficulty": "hard", 
    "total_questions": 3,
    "expires_in_days": 14
  }
  ```
- **Returns:** `{ "interview_id": "8b89c375-...", "shareable_link": "..." }`

---

## 🧑‍💻 3. AI Inference & Interview Engine (`/api/sessions`)
*This is the mathematical core of the platform. It handles LLM contextual generation, PyPDF parsing, and asynchronous ML inference execution.*

### `POST /api/sessions/start` ⚠️ **(Multipart Form Data)**
- **Internal Logic (Heavy Pipeline):**
  1. Checks local state to enforce the **Anti-Duplicate Data Lock**.
  2. If a PDF is attached, physical buffers are streamed into the `PyPDF2` engine for raw text extraction.
  3. The extracted text is injected procedurally into a highly specialized LangChain System Prompt.
  4. Triggers `gemini-2.5-flash`, forcing it to adhere to a strict Pydantic JSON array schema.
  5. Stores the array locally into the `sessions` DB to ensure 0ms latency for the frontend.
- **Form Fields:** 
  - `interview_id`: (string) e.g., "8b89c375-..."
  - `candidate_name`: (string) e.g., "Zain Jadoon"
  - `candidate_email`: (string) e.g., "zain@example.com"
  - `resume`: *(File Upload - .pdf | Optional Context Injector)*
- **Returns:** `{ "session_id": "...", "all_questions": ["Q1", "Q2", "Q3"] }`

### `POST /api/sessions/submit_all`
- **Internal Logic (Asynchronous Engine):**
  1. Absorbs the frontend React window analytics (Tab switching counts, elapsed time counters).
  2. Commits a strict state lock on MongoDB (`status: analyzing`).
  3. Initiates a non-blocking `FastAPI BackgroundTask`. While the frontend immediately receives a "Thank You" loading screen, the server internally boots up **DistilBERT** to tokenize the answers into multi-dimensional vectors, calculates similarity mapping, and then checks findings against Gemini 2.5 Flash for hallucination fallback safety.
- **Request Body (`application/json`):** 
  ```json
  {
    "session_id": "paste_session_id_here",
    "answers": ["Answer 1", "Answer 2", "Answer 3"],
    "cheating_flags": {
      "tab_switches": 2,
      "copy_paste_attempts": 0,
      "time_taken_seconds": 315
    }
  }
  ```

---

## 📊 4. Analytical Data Aggregation (`/api/reports`)
*These endpoints translate the complex AI JSON outputs into easily digestible metrics for the HR Dashboard tables.*

### `GET /api/reports/candidates` 🔒 *(Protected)*
- **Internal Logic:** Executes a high-performance MongoDB cursor, excluding massive Q&A arrays over the network, to quickly provide a table-ready list of all candidates alongside their AI-graded percentage.
- **Returns:** `[{ "candidate_name": "Zain Jadoon", "is_passed": true, "final_grade_percentage": 90.0, "status": "completed" }, ...]`

### `GET /api/reports/candidate/{session_id}` 🔒 *(Protected)*
- **Internal Logic:** Reconstructs the exact session logic. Returns the complete AI payload, including the "Model Difference Chart" which allows recruiters to compare exactly how the Local DistilBERT model rated the answer versus how the massive remote Google model rated the answer.
- **Returns:** Massive JSON breakdown required for the Candidate Result interface.
