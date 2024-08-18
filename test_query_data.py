from query_data import query_rag
from get_embedding_function import get_embeddings
from langchain_openai import OpenAI
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_aws_notes():
    assert query_and_validate(
        question = "What is Amazon Elastic File System?",
        expected_response= "It is a serverless network file system used for sharing files."
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response = expected_response, actual_response = response_text
    )

    model = OpenAI()
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    evaluator = load_evaluator("embedding_distance", embeddings = get_embeddings(), distance_metric = EmbeddingDistance.EUCLIDEAN)

    if "true" in evaluation_results_str_cleaned:
        # Print response in green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Get the pairwise distance between the expected result and actual result embeddings 
        prob_result = evaluator.evaluate_strings(
            prediction = response_text, reference = expected_response
        ) 
        if prob_result['score'] >= 0.8:
            print("\033[92m" + f"Response: True" + "\033[0m")
            return True
        else:
            #Print response in red if incorrect
            print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return False
    else:
        raise ValueError(f"Invalid evaluation result. Cannot determine if 'true' or 'false'.")