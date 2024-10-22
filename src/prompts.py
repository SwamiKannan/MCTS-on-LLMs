import re


seed_answers = [
    "I don't know the answer",
    'I cannot say',
    "I'm not sure"
]

def chat_completion_request(prompt):
    messages = [{'role':'user', 'content':prompt}]
    return messages
    # return the content of LLM response with temperature 1.0 and max_tokens = 1500

def get_critique(question, draft_answer):
    prompt = (f'Question: {question}\n',
              f'Draft answer: {draft_answer}\n'
              'Please critique the draft answer'
              'Do a careful assessment of whether the answer is correct or not, and why',
              'Consider multiple ways of verifying the correctness of the answer',
              'Do point out every flaw and hold the draft answer to a high standard',
              'Do provide specific recommendations to improve the answer',
              'Do think step by step',
              'Do not provide a revised answer'
    )

    return chat_completion_request(prompt)

def improve_answer(question, draft_answer, critique):
    prompt = (
        f'Question: {question}\n'
        f'Answer: {draft_answer}\n'
        f'Critique: {critique}\n',
        'Please improve the draft answer based on the critique. Follow this format'
        'Reasoning process: <step-by-step reasoning process>\n'
        'Verification: <verification of the facts>\n'
        'Final Answer: <the improved and verified answer>\n'
    )
    return chat_completion_request(prompt)

def rate_answer(question, answer):
    prompt = (
        f'Question: {question}\n'
        f'Answer: {answer}\n\n'
        "Assess whether the user's query requires only the final answer or whether a justification is required"
        'As an expert on this topic, please suggest a detailed critique of the answer, pointing out every flaw'
        'Provide only a critique, not a suggested answer'
        'Then, rate the answer on a scale of 0 to 100'
        'The response should be in the following format:\n',
        'Critique: <detailed critique>\n'
        'Rating: <rating>'
        )
    
    rating_response = chat_completion_request(prompt)
    try:
        match = re.search(r'Rating:\s*(\d+)', rating_response)
        if match:
            rating  = int(match.group(1))
            if rating > 95:
                rating = 95
            rating = float(rating)/100
        else:
            raise ValueError('Rating not found in the response')
    except Exception  as e:
        print(f'Error in extracting rating: {e}')
        print(f'Rating response was: {rating_response}')
        rating = 0.0
    return rating

def get_answer_directly_from_llm(question):
    prompt = (
        f"Question {question}\n"
        "Please provide the answer with detailed reasoning. Follow the format:\n"
        "Reasoning process: <step-by-step reasoning process>\n"
        "Verification: <verification of the facts>\n"
        "Final answer: <the improved and verified answer>\n"

    )
    direct_llm_response = chat_completion_request(prompt)

    try:
        match = re.search(r'Final answer:\s*(.*)', direct_llm_response)
        final_answer = match.group(1).strip() if match else None
    except Exception as e:
        final_answer = None
    return direct_llm_response, final_answer
