import requests
import random
import json
import nltk

BASE_URL_QG = "http://127.0.0.1:8010"
BASE_URL_AG = "http://127.0.0.1:8011"
BASE_URL_DG = "http://127.0.0.1:8012"


def load_sentence_tokenizer():
    return nltk.data.load('tokenizers/punkt/russian.pickle')

def generate_answer(context):
    data = {
        "context": context.replace("\n", " ")
    }
    response = requests.post(f"{BASE_URL_AG}/generate", json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response, "error_code": response.status_code}


def generate_question(answer, context):
    data = {
        "answer": answer.replace("\n", " "),
        "context": context.replace("\n", " ")
    }
    response = requests.post(f"{BASE_URL_QG}/generate", json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response, "error_code": response.status_code}


def generate_distractors(question, answer, context):
    data = {
        "question": question.replace("\n", " "),
        "answer": answer.replace("\n", " "),
        "context": context.replace("\n", " ")
    }

    response = requests.post(f"{BASE_URL_DG}/generate", json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response, "error_code": response.status_code}


def generate_test_question(context):
    response = generate_answer(context)
    if "error_code" in response:
        return response
    answer = response["answer"]
    response = generate_question(answer, context)
    if "error_code" in response:
        return response
    question = response["question"]
    response = generate_distractors(question, answer, context)
    if "error_code" in response:
        return response
    distractors = response["distractors"]

    answers = distractors + [answer]

    random.shuffle(answers)
    correct_answer_id = answers.index(answer)

    return {
        "question": question,
        "answers": answers,
        "correct_answer_text": answer,
        "correct_answer_id": correct_answer_id
    }
