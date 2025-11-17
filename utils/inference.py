import json
from utils.llm import LLM
from langchain_core.prompts import PromptTemplate
import re
from requests.exceptions import HTTPError

def build_prompt_template() -> PromptTemplate:
    template = """
    ## IELTS Writing Test Information
    The IELTS Writing test evaluates your ability to express yourself in written English. The writing sample must address the {writing_question} and adhere to the following requirements:

    ### Test Format
    - The IELTS Writing test consists of two tasks:
      - **Task 1**:
        - Academic: Describe visual information, such as graphs, charts, or diagrams.
        - General Training: Write a letter based on a given situation.
      - **Task 2**: Write an essay in response to a question or statement.

    ### Word Count
    - **Task 1**: Minimum 150 words.
    - **Task 2**: Minimum 250 words.

    ### Word Count Impact
    Responses below the recommended word count are likely to lack sufficient depth and development of ideas, negatively affecting the **Task Response** score. Deduct from the overall score like this:
    - **Task 1**:
      - 100–149 words: Deduct 1 band.
      - Below 100 words: Deduct 2 bands.
    - **Task 2**:
      - 200–249 words: Deduct 1 band.
      - 150–199 words: Deduct 2 bands.
      - Below 150 words: Deduct 3 bands.

    ## Scoring Instructions
    Evaluate the writing sample on a scale from 1 to 9 using the following steps:

    ### 1. Task Type Context
    - Determine whether the task is Task 1 or Task 2 based on the {writing_question}, and evaluate accordingly.

    ### 2. Task Response
    - Assess how well the response addresses the question, develops ideas, and supports them with examples or evidence.
    - Penalize for insufficient word count based on the **Word Count Impact** section.

    ### 3. Coherence and Cohesion
    - Evaluate the logical flow, organization of ideas, and use of cohesive devices.

    ### 4. Lexical Resource
    - Examine the range, accuracy, and appropriateness of vocabulary used.

    ### 5. Grammatical Range and Accuracy
    - Analyze the sentence structures, grammar, and punctuation for range and accuracy.

    ## Additional Context
    - If the {writing_sample} is completely off-topic from the given {writing_question}, do not assign a score. Instead, provide reasoning for this decision.

    # IELTS Writing Evaluation

    ## Writing Sample
    {writing_sample}

    ## Question
    {writing_question}

    ## Evaluation Criteria
    {criteria_string}

    ## Test Variant
    {test_variant}


    ## JSON Response Format
    The correct word count has already been computed as {word_count}.  
    You MUST copy this exact value into the "word_count" field in the JSON and MUST NOT recalculate it yourself.
    
    Provide the response as **valid JSON only**, with no preamble or explanation outside the JSON.
    Use this structure:
    {{
        "score": 1,
        "reasoning": "Detailed explanation for the score, referencing specific points from the writing sample and evaluation criteria.",
        "test_variant": "{test_variant}",
        "word_count": {word_count},
        "misspelled_words": []
    }}
    Ensure the response is a single JSON object with opening and closing curly braces.
    """

    return PromptTemplate.from_template(template)



def invoke_llm(prompt_template: PromptTemplate, writing_sample: str = "", writing_question: str = "", criteria_string: str = "", test_variant: str = "", word_count: int = 0) -> dict:
    print(f"Invoking LLM with prompt template... {word_count}", flush=True)
    """
    Invokes the LLM with the provided prompt template and input data.

    Args:
        prompt_template (PromptTemplate): The prompt template for the LLM.
        writing_sample (str, optional): The writing sample to evaluate. Default is "".
        writing_question (str, optional): The writing question to evaluate. Default is "".
        criteria_string (str, optional): The evaluation criteria. Default is "".
        test_variant (str, optional): The test variant (Academic or General Training). Default is "".
        word_count (int, optional): The word count of the writing sample. Default is 0.

    Returns:
        dict: The processed response from the LLM, or a structured error dictionary if invocation fails.
    """
    llm = LLM().get_groq_llm()
    chain_evaluation = prompt_template | llm

    try:
        response = chain_evaluation.invoke({
            "writing_sample": writing_sample,
            "writing_question": writing_question,
            "criteria_string": criteria_string,
            "test_variant": test_variant,
            "word_count": word_count
        })

        if not response or not hasattr(response, 'content') or not response.content:
            return {
                "error": "LLM did not return a valid response. Possibly due to usage limits or a network issue.",
                "code": "NO_CONTENT"
            }

        response_text = response.content.strip()
        response_text = re.sub(r"^```(?:json)?", "", response_text, flags=re.IGNORECASE).strip()
        response_text = re.sub(r"```$", "", response_text).strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "warning": "Response was not valid JSON. Returning raw output.",
                "content": response_text
            }

    except HTTPError as http_err:
        if http_err.response.status_code == 429:
            return {
                "error": "LLM usage limit exceeded. Please wait or upgrade your quota.",
                "code": "LIMIT_EXCEEDED"
            }
        else:
            return {
                "error": f"HTTP error occurred: {str(http_err)}",
                "code": "HTTP_ERROR"
            }
    except Exception as e:
        return {
            "error": f"Unexpected error during LLM invocation: {str(e)}",
            "code": "INTERNAL_ERROR"
        }


def classify_writing_task(writing_question: str) -> str:
    classification_prompt = PromptTemplate.from_template(
        f"""
        ### QUESTION:
        {writing_question}

        ### INSTRUCTION:
        Based on the question above, determine whether the task is 'Academic' or 'General Training'.
        - If the question asks to describe a graph, chart, or diagram, classify it as 'Academic'.
        - If the question asks to write a letter or is conversational in tone, classify it as 'General Training'.

        ### RESPONSE FORMAT:
        Return only one word: 'Academic' or 'General Training'.
        """
    )

    result = invoke_llm(classification_prompt)
    classification = result.get("content", "").strip()

    if classification not in ["Academic", "General Training"]:
        raise ValueError(f"Invalid classification response: {classification}")

    return classification

def evaluate(writing_sample: str, writing_question: str, criteria_string: str, test_variant: str, word_count: int) -> dict:
    """
    Main function to evaluate the writing sample against the criteria.

    Args:
        writing_sample (str): The writing sample provided by the user.
        writing_question (str): The question to evaluate.
        criteria_string (str): The evaluation criteria.

    Returns:
        dict: The evaluation results as a dictionary.
    """
    prompt_template = build_prompt_template()
    return invoke_llm(prompt_template, writing_sample, writing_question, criteria_string, test_variant, word_count)
