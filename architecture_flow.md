# System Architecture & Flow: Multi-Tenant AI Interview Platform

**Project:** Federated AI Interview Agent (S26-067-R)
**Current Stage:** Implementing MVP conventional architecture (SaaS)

---

## 1. System Overview

The product is a multi-tenant SaaS platform that enables organizations to create, distribute, and evaluate AI-driven interviews through shareable links. 

The system relies on a specialized, decoupled architecture:
- **Frontend Interfaces:** Next.js (React) for both the Recruiter Dashboard and Candidate Interview interface.
- **Backend APIs:** FastAPI to handle logic, routing, and serve as the AI host.
- **Persistent Storage:** MongoDB to store templates, sessions, and multi-tenant data.
- **Question Generation & Orchestration:** Google Gemini API orchestrated by **LangGraph** (Stateful AI execution).
- **Response Evaluation:** Our locally fine-tuned **DistilBERT Classification Model** (v2).

---

## 2. Functional Modules

### 2.1 Company Authentication & Tenant Isolation
**Purpose:** Enable organizations to securely access and manage their own interview workflows without data leakage.
- **Core Functions:** User registration, login, JWT-based authentication.
- **Multi-tenant Logic:** Every database query mapping to a company must strictly include `{"company_id": current_company}`.
- **Security:** Password hashing via `bcrypt`, role-based access (admin, recruiter).

### 2.2 Interview Template Creation Engine
**Purpose:** Allow companies to define interview parameters that drive AI behavior.
- **Input Parameters:** Role (e.g., Software Engineer), Skills (e.g., React, Python), Difficulty Level (Easy/Medium/Hard), Duration / Max Questions.
- **Processing:** Validates input, stores configuration, generates a unique `interview_id`.

### 2.3 Secure Link Generation System
**Purpose:** Provide controlled access to candidates via unique URLs.
- **Mechanism:** Generates a UUID for the interview template.
- **Format Example:** `/interview/{interview_id}`
- **Future Enhancements:** Expiry timestamps, attempt limits, signed HMAC URLs.

### 2.4 Interview Session Management
**Purpose:** Track each candidate interaction dynamically and independently.
- **Key Concept:** We separate the **Interview Template** (static definition) from the **Interview Session** (dynamic candidate run).
- **Session Creation:** Triggered automatically when a candidate opens the share link and enters their name.
- **Stored Fields:** `session_id`, `interview_id`, candidate metadata, and the current LangGraph state.
- **Lifecycle:** Created → Active → Completed → Analyzed.

### 2.5 AI Interview Orchestration Engine (LangGraph)
**Purpose:** Drive the entire interview dynamically using stateful logic.
- **Nodes in Graph:**
  1. `generate_question` (Uses Gemini API to ask a question based on candidate state and resume)
  2. `evaluate_answer` (Sends answer to DistilBERT API)
  3. `adjust_difficulty` (State logic mutation)
  4. `end_interview` (Termination)
- **State Object:** 
  ```python
  class InterviewState:
      question: str
      answer: str
      score: int
      history: list
      difficulty: str
      skills: list
  ```

### 2.6 Response Evaluation Engine (DistilBERT)
**Purpose:** Assess candidate answers with high accuracy and consistency.
- **Scoring Layer:** Our fine-tuned DistilBERT classifier.
- **Output:** Categorized numerical score (0 = Poor, 1 = Average, 2 = Excellent). 
- *Note:* Gemini can be used as an optional secondary layer to provide textual qualitative feedback to the candidate alongside the hard DistilBERT score.

### 2.7 Adaptive Questioning System
**Purpose:** Simulate real interview dynamics by adjusting difficulty.
- **Logic Matrix:** LangGraph reads the DistilBERT score. If the candidate consistently scores `2` (Excellent), the state mutates the difficulty to "Hard" for Gemini's next prompt. If the candidate scores `0`, the difficulty drops to "Easy".

### 2.8 Data Persistence Layer (MongoDB)
**Collections:**
1. **Companies:** `company_id`, `name`, `users`, `auth_hash`
2. **Templates (Interviews):** `interview_id`, `company_id`, `config` (role, skills)
3. **Sessions:** `session_id`, `interview_id`, `candidate_data`, `transcript` (array of Q&A with scores), `timestamps`

### 2.9 Analytics & Reporting Engine
**Purpose:** Provide actionable insights to recruiters via the Next.js Dashboard.
- **Metrics:** Average candidate score, completion rate.
- **Output:** Next.js Dashboard UI, Exportable PDF reports (via `jsPDF`).

---

## 3. End-to-End Workflow

1. **Company Initialization:** Recruiter creates an interview template mapping to a role and skills.
2. **Link Sharing:** System generates `/interview/{id}` link.
3. **Candidate Entry:** Candidate opens the link and starts the session.
4. **LangGraph Loop:**
   - Gemini generates the first question based on the template.
   - Candidate responds via UI.
   - Answer is passed to DistilBERT for scoring (0, 1, 2).
   - LangGraph adapts difficulty context based on DistilBERT score.
5. **Termination:** Upon reaching question limit, interview ends. Results are persisted to MongoDB.
6. **Reporting:** Dashboard automatically updates with candidate results.

---

## 4. MVP Scope Definition

To avoid overengineering, the **Initial MVP** focuses purely on:
1. Interview Template creation
2. Shareable link generation
3. Basic AI LangGraph loop (Gemini questions, DistilBERT scoring)
4. MongoDB Storage
5. Simple Next.js Recruiter Dashboard

### Expansion Roadmap (Post-MVP)
- **Phase 2:** Voice processing pipeline (Whisper STT / ElevenLabs TTS), Anti-cheating controls (tab switching tracking).
- **Phase 3:** Multi-agent LangGraph architectures, Advanced analytics.
- **Phase 4:** Federated Learning Integration (Stage 4 & 5 of FYP scope).
