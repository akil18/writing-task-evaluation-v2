from fastapi import FastAPI
from pydantic import BaseModel
from utils.inference import evaluate
from utils.inference import classify_writing_task
from utils.criteria import get_criteria
from utils.tools import word_count_checker
import gradio as gr
from app.gradio_app import gradio_app

app = FastAPI()

class Request(BaseModel):
    writing_sample: str
    writing_question: str
    task_type: int

class Response(BaseModel):
    evaluation: dict

@app.post("/evaluate", response_model=Response)
async def evaluate_api(prompt: Request):
    criteria_string = get_criteria(prompt.task_type)
    test_variant = classify_writing_task(prompt.writing_question)
    word_count = word_count_checker(prompt.writing_sample)
    try:
        response = evaluate(prompt.writing_sample, prompt.writing_question, criteria_string, test_variant, word_count)
        return {"evaluation": response}
    except Exception as e:
        return {"error": str(e)}

# Create the Gradio interface
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
