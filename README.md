# 🎓 College Admission Agent

> **IBM AICTE Internship 2026 — Problem Statement No. 4**
> A RAG-based AI assistant that helps students with college admission queries.

---

All college names, fees, eligibility criteria, dates, and scholarship details in the
`knowledge_base/` folder are **fictitious sample data** created purely for demonstration.
Do not use this information for actual college admissions.

---

## What This Application Does

The College Admission Agent is a **Retrieval-Augmented Generation (RAG)** chatbot that:

| Feature | Description |
|---|---|
| 🔍 Course selection | Explains available programmes and helps choose the right one |
| ✅ Eligibility | Answers eligibility criteria for each programme |
| 📝 Application process | Guides students through the step-by-step application |
| 📄 Required documents | Lists all documents needed for admission |
| 💰 Fee structure | Provides accurate fee details from the knowledge base |
| 🏆 Scholarships | Describes available scholarships and how to apply |
| 📅 Deadlines | States all admission deadlines clearly |
| ❓ FAQs | Answers frequently asked admission questions |

**Crucially**, the agent will say *"I could not find this information in the admission knowledge base"*
instead of guessing when a question falls outside the provided documents.

---

## Architecture

```
Student Question
      │
      ▼
 Streamlit UI (app.py)
      │
      ▼
 Admission Agent (admission_agent.py)
      │
      ├──► RAG Retriever (rag/retriever.py)
      │         │
      │         ├──► TF-IDF Vector Store (rag/vector_store.py)
      │         │
      │         └──► Knowledge Base (knowledge_base/*.txt)
      │
      └──► IBM Granite 3.3 via watsonx.ai
                │
                ▼
         Grounded Answer + Source References
```

**RAG flow:**
1. The student's question is converted to a TF-IDF vector.
2. Cosine similarity search finds the top-3 most relevant knowledge-base chunks.
3. Those chunks are injected into the IBM Granite prompt as grounding context.
4. IBM Granite generates an answer strictly based on the provided context.
5. Source file names and relevance scores are shown to the student.

---

## Project Structure

```
College_Admission_Agent/
├── app.py                        # Streamlit UI
├── admission_agent.py            # Agent orchestration (RAG + LLM)
├── rag/
│   ├── __init__.py
│   ├── retriever.py              # KB loader, text chunker, retrieve()
│   └── vector_store.py           # TF-IDF vector store
├── knowledge_base/
│   ├── application_process.txt   # Step-by-step application guide
│   ├── courses.txt               # Programme catalogue
│   ├── deadlines.txt             # Admission deadlines
│   ├── documents.txt             # Required documents checklist
│   ├── eligibility.txt           # Eligibility criteria
│   ├── faq.txt                   # Frequently asked questions
│   ├── fees.txt                  # Fee structure
│   └── scholarships.txt          # Scholarship programmes
├── config.py                     # IBM watsonx.ai config (reads .env)
├── requirements.txt
├── .env.example                  # Template — copy to .env and fill in
└── README.md
```

---

## Prerequisites

- Python 3.10 or later
- An IBM Cloud account with access to **IBM watsonx.ai**
- A watsonx.ai project with the **IBM Granite** model enabled

---

## Setup Instructions

### 1. Clone / open the project

```bash
cd College_Admission_Agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM watsonx.ai credentials

Copy the example file and fill in your credentials:

```bash
copy .env.example .env       # Windows
# or
cp .env.example .env          # macOS / Linux
```

Edit `.env`:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

**How to find your credentials:**
- `WATSONX_API_KEY` → IBM Cloud → Manage → Access (IAM) → API Keys → Create
- `WATSONX_PROJECT_ID` → watsonx.ai platform → your project → Manage → General → Project ID
- `WATSONX_URL` → depends on your region; default is `https://us-south.ml.cloud.ibm.com`

> **Note:** If credentials are not configured, the app still runs and shows retrieved context,
> but AI-generated answers will be disabled with a clear message.

### 5. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Using the Application

1. Type your admission question in the chat box at the bottom of the page.
2. Or click one of the **suggested questions** shown on the welcome screen.
3. The agent will display the answer and, under **"📖 Sources used"**, show which
   knowledge-base file(s) it retrieved the answer from.
4. If the knowledge base does not contain the answer, the agent will say so clearly.

---

## Extending the Knowledge Base

To add or update admission information:

1. Open any file in `knowledge_base/` (or create a new `.txt` file).
2. Add your content in plain text. Sections separated by blank lines become natural chunk boundaries.
3. Restart the Streamlit app. The RAG index rebuilds automatically on startup.

No code changes are needed to extend the knowledge base.

---

## Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | IBM Granite 3.3 8B Instruct via IBM watsonx.ai |
| RAG retrieval | TF-IDF cosine similarity (scikit-learn) |
| Knowledge base | Plain `.txt` files |
| Environment config | python-dotenv |
| Language | Python 3.10+ |

---

## IBM Granite Model Details

- **Model ID:** `ibm/granite-3-3-8b-instruct`
- **Temperature:** `0.0` (deterministic output reduces hallucinations)
- **Max new tokens:** 512
- The model is instructed via system prompt to answer **only** from the provided context
  and to say it cannot find the information if the context does not contain the answer.

---

## Key Design Decisions

- **No external vector database** — TF-IDF with cosine similarity is sufficient for a small,
  static knowledge base and keeps the project beginner-friendly with no extra services.
- **Hallucination guard** — Temperature = 0, explicit system prompt rules, and a fallback
  "not found" response when retrieval score is below threshold.
- **Modular structure** — UI, agent, and retrieval are in separate files so each can be
  understood and modified independently.
- **Plain-text knowledge base** — Easy to read, extend, and understand without a database.

---

## License

This project is created for the **IBM AICTE Internship 2026** and is intended for
educational and demonstration purposes only.
