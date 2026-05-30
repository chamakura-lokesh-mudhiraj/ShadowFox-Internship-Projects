# ShadowMind Nexus AI

## Advanced Language Intelligence & Research Platform

ShadowMind Nexus AI is a portfolio-level AI system built from the ShadowFox Task 3 brief. It is not only a notebook that tests a model. It is an interactive language intelligence platform that evaluates, analyzes, and visualizes LLM behavior using modern NLP pipelines, LangChain-style workflows, memory, document retrieval, analytics, and research dashboards.

## Project Vision

ShadowMind Nexus AI demonstrates the mindset of an AI engineer and research analyst:

- Deploy language models and study their behavior.
- Compare prompts, model confidence, and response latency.
- Build memory-based AI chat.
- Retrieve answers from private documents using a RAG-style flow.
- Visualize results through an analytics dashboard.
- Persist audit events and export prompt logs for review.
- Configure workspace metadata, analyst name, use case, and risk level.
- Explain limitations, ethics, and future improvements.

## Official Task Alignment

| Guideline | Where It Is Covered |
| --- | --- |
| Problem statement | `PROJECT_REPORT.md`, `reports/RUBRIC_ALIGNMENT.md`, notebook introduction |
| LM selection | DistilBERT for understanding, DistilGPT2 for optional generation |
| Jupyter Notebook implementation | `ShadowMind_LM_Analysis.ipynb` |
| Exploration and analysis | `src/shadowmind_analysis.py`, Research Lab, notebook experiments |
| Research questions | Notebook and `PROJECT_REPORT.md` |
| Visualization | Notebook charts and app analytics |
| Evaluation and ethics | Notebook, report, and risk notes in the app |
| Conclusion and insights | Notebook conclusion and `PROJECT_REPORT.md` |

## System Architecture

```text
User Interface
  -> Authentication Layer
  -> Prompt Engine
  -> LangChain-Oriented Core
  -> Multiple LLM Support
  -> Memory + Context
  -> RAG / Knowledge Base
  -> Response Generator
  -> Analytics + Visualization
  -> Research Dashboard
```

## Platform Modules

| Module | Purpose |
| --- | --- |
| Multi-LLM Intelligence Hub | Compare local reasoning and HuggingFace model behavior |
| Prompt Engineering Studio | Test simple, professional, teacher, and research prompt styles |
| AI Memory Chatbot | Store session facts and use them in later responses |
| Research & Evaluation Lab | Benchmark sentiment, ambiguity, sarcasm, context, and domains |
| AI Analytics Dashboard | Track prompts, confidence, latency, model usage, task mix, and audit events |
| Sentiment + Emotion Intelligence | Analyze emotional direction and difficult language |
| RAG Knowledge System | Retrieve context from uploaded notes and sample documents with logged traces |
| AI Explainability System | Show confidence, keyword-based evidence, risk notes, and limitations |
| Workspace Settings | Configure analyst/project context and clear runtime logs |

## Folder Structure

```text
app/
  ui/              # Theme and product layout
  pages/           # Streamlit pages
  components/      # Charts and reusable UI elements
  auth/            # Future authentication module
  llm/             # Model adapters and benchmark engine
  rag/             # Document retrieval system
  analytics/       # Metrics and leaderboard logic
  memory/          # Conversation memory
  storage/         # JSONL audit logs, settings, and CSV exports
notebooks/         # Research notebook area
datasets/          # Sample knowledge base and datasets
vector_db/         # Placeholder for future embeddings/vector store
reports/           # Final reports and architecture documents
images/            # Screenshots and visuals
tests/             # Unit tests
src/               # Original reusable research code
runtime/           # Local generated logs, ignored by Git
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Interactive App

```bash
streamlit run app/main.py
```

## Run Research Notebook

Open `ShadowMind_LM_Analysis.ipynb` in Jupyter and run all cells.

```bash
jupyter notebook
```

## Run Tests

```bash
pytest
```

## Real-World Features Added

- Persistent JSONL audit trail in `runtime/events.jsonl`.
- Downloadable prompt log CSV from the Analytics Dashboard.
- Workspace configuration for project name, analyst, use case, and risk level.
- Prompt run records with model, task type, confidence, latency, evidence, and risk notes.
- Document retrieval events with query, sources, and result count.
- Unit tests for memory, retrieval, and persistence.

## Model Support

The app supports two modes:

- `Nexus Local Reasoner`: offline fallback for demos when ML packages are unavailable.
- `HuggingFace DistilBERT`: Transformer sentiment model using `distilbert-base-uncased-finetuned-sst-2-english`.
- `DistilGPT2`: optional text-generation model used in the research code for generation capability analysis.

The fallback mode keeps the UI and workflow usable even before `transformers` and `torch` are installed. For final submission screenshots, install all requirements and run the HuggingFace mode.

## Submission Checklist

- `ShadowMind_LM_Analysis.ipynb` for the required Jupyter notebook.
- `PROJECT_REPORT.md` for formal explanation and findings.
- `reports/RUBRIC_ALIGNMENT.md` for direct guideline mapping.
- `app/main.py` for the real-world interactive platform.
- `tests/` for verification of memory, retrieval, and persistence modules.

## Final Project Statement

ShadowMind Nexus AI is an advanced AI-powered Language Intelligence and Research Platform that integrates LangChain-oriented workflows, language model experimentation, memory-based conversational AI, document intelligence, analytics dashboards, and NLP evaluation to analyze and visualize language model performance in real-world scenarios.
