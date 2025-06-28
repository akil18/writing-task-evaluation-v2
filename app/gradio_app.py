import gradio as gr
from utils.inference import evaluate
from utils.inference import classify_writing_task
from utils.criteria import get_criteria
from utils.tools import word_count_checker

def gradio_interface_fn(writing_question, writing_sample, task_type):
    criteria_string = get_criteria(task_type)
    test_variant = classify_writing_task(writing_question)
    word_count = word_count_checker(writing_sample)
    return evaluate(writing_sample, writing_question, criteria_string, test_variant, word_count)

# Create the Gradio interface
gradio_app = gr.Interface(
    fn=gradio_interface_fn,
    inputs=[
        gr.Textbox(label="Question", lines=6),
        gr.Textbox(label="Answer", lines=6),
        gr.Dropdown(label="Task Type", choices=[1, 2]) 
    ],
    outputs=gr.Textbox(label="Output"),
    title="Writing Test Evaluation App",
    description="Enter the question you attempted and the answer in their respective boxes.",
)
