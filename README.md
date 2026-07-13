# IntelliMoE — Multi-Expert AI Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Frontend Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Backend Routing](https://img.shields.io/badge/Architecture-Mixture--of--Experts-4F8CFF?style=flat-square)](https://github.com/suresh4330/IntelliMoE)
[![Database ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-00C853?style=flat-square)](https://www.trychroma.com/)

IntelliMoE is a premium, multi-expert AI assistant built on a Mixture of Experts (MoE) routing engine. It dynamically routes user prompts to specialized LLM experts (Coding, ML, Math, Deep Learning, Research, System Design, GenAI) based on semantic classification, evaluates the output with an Answer Quality Engine (planning, review, improvement), and presents diagnostic explainability metrics in a dark-themed UI.

---

## 🚀 Key Features

* **🧠 Semantic MoE Routing**: Routes user prompts dynamically to 7 specialized experts (Coding, Math, ML, Deep Learning, Research, System Design, GenAI) using a hybrid classifier (Machine Learning TF-IDF + LLM fallback).
* **✨ Answer Quality Engine**: Enhances every output through a systematic multi-stage pipeline:
  * `ResponsePlanner`: Generates an answer structure before generation.
  * `ResponseReviewer`: Audits correctness, completeness, clarity, and formatting.
  * `ResponseImprover`: Resolves review critiques to output only the highest quality answers.
* **🎙️ Voice Typing (ChatGPT Style)**: Integrates the browser's Web Speech API for voice-to-text. Features an active recording pulsing state `🔴` and React state-binding bypass for auto-submission.
* **🔊 Text-to-Speech Voice Mode**: Auto-reads responses aloud when a user dictates a query. Includes interactive inline speaker controls `🔊/🔊x` next to each message to play/stop playback anytime.
* **🔍 Explainable AI (XAI)**: Diagnostic developer panels displaying active routing logic, plan traces, review criteria, performance latency benchmarking, and raw JSON schemas.
* **🔬 Research RAG Pipeline**: Integrates ChromaDB and `SentenceTransformers` to index, retrieve, and inject context from academic research papers.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/suresh4330/IntelliMoE.git
cd IntelliMoE
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run ui/app.py
```
The application will open automatically at `http://localhost:8502/`.

---

## 📂 Project Structure

```
IntelliMoE/
├── config/              # Configuration & settings management
├── conversation_ai/     # Conversational AI layer (persona, human-like prompts)
├── experts/             # Domain-specific expert LLM wrappers
├── explainability/      # Diagnostics, benchmarking, and XAI telemetry
├── prompts/             # Expert prompts and instructions
├── router/              # MoE routing engine & Answer Quality Engine
├── services/            # Client connectors for Gemini and Groq APIs
├── tests/               # Pytest suite
├── ui/                  # Streamlit frontend application code
└── utils/               # Memory, logging, feedback, and vector stores
```

---

## 🧪 Running Tests
Run the pytest suite to verify local execution and api connectivity:
```bash
pytest tests/ -v
```
