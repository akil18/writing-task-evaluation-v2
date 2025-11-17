from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.inference import evaluate, classify_writing_task
from utils.criteria import get_criteria
from utils.tools import word_count_checker
import gradio as gr
from app.gradio_app import gradio_app

app = FastAPI()

# 🔐 CORS setup
origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
    "https://writing-task-app.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧾 Request & Response Models
class Request(BaseModel):
    writing_sample: str
    writing_question: str
    task_type: int

class Response(BaseModel):
    evaluation: dict

# 📬 POST /evaluate endpoint
@app.post("/evaluate", response_model=Response)
async def evaluate_api(prompt: Request):
    criteria_string = get_criteria(prompt.task_type)
    test_variant = classify_writing_task(prompt.writing_question)
    word_count = word_count_checker(prompt.writing_sample)
    print(f"Evaluating writing sample with word count: {word_count}", flush=True)
    try:
        response = evaluate(
            prompt.writing_sample,
            prompt.writing_question,
            criteria_string,
            test_variant,
            word_count
        )
        print( "Evaluation response:", response, flush=True)
        return {"evaluation": response}
    except Exception as e:
        return {"evaluation": {"error": str(e)}}

# 🚦 Health Check Endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# 🎛 Mount Gradio at /gradio
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
