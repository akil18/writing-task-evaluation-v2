# IELTS Writing Evaluation Service

This repository contains a **Python FastAPI service** that evaluates IELTS writing tasks using an LLM (via LangChain). It’s designed to be the “evaluation brain” of a larger app: you send it a writing sample and question, and it returns a structured JSON evaluation.

---

## ✨ Features

- REST API endpoint: `POST /evaluate`
- Uses LangChain + your configured LLM
- Provides structured IELTS scoring:
  - score (1–9)
  - reasoning
  - detects test variant
  - provides word count
  - gives a list of misspelled words
- Fully JSON-only responses
- Optional Gradio UI at `/gradio`

---

## 🧱 Tech Stack

- **Python 3.x**
- **FastAPI + Uvicorn**
- **LangChain**
- **Groq**
- **Pydantic**
- **Gradio**

---

## 🧪 Running Locally

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd writing-task-evaluation-v2
```

### 2. Create & activate a virtual environment

Windows (Git Bash):

```bash
python -m venv venv
source venv/Scripts/activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a .env file

```bash
WTE_GROQ_API_KEY=your-key-here
MODEL=chosen-model-from-groq
```

### 5. Start FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🔌 API Endpoints

### POST /evaluate

Evaluates an IELTS writing sample.

Request:

```bash
{
  "writing_sample": "Some IELTS writing sample...",
  "writing_question": "Some Task 1 or Task 2 question...",
  "task_type": 2
}
```

Response:

```bash
{
  "evaluation": {
    "score": 7,
    "reasoning": "The answer addresses both views...",
    "test_variant": "Academic",
    "word_count": 285,
    "misspelled_words": []
  }
}
```

### Route /gradio

Opens a browser UI for testing

### GET /health

For status of server

```bash
{"status": "ok"}
```
