from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer
)
import torch

SOURCE_MAX_TOKEN_LEN = 384
TARGET_MAX_TOKEN_LEN = 80
MODEL_NAME = "qg_model"
#MODEL_NAME = "qg_model_mt5_base_new"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QGModel:
    def __init__(self, model_name):
        self.tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
        self.model = MT5ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            return_dict=True,
            device_map='auto')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qg_model
    qg_model = QGModel(MODEL_NAME)
    yield
    # Shutdown ops to be placed here


class GenerateRequest(BaseModel):
    answer: str
    context: str


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: GenerateRequest):
    source_encoding = qg_model.tokenizer(
        'answer: {} context: {}'.format(request.answer, request.context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qg_model.model.generate(
        input_ids=source_encoding['input_ids'].to(device),
        attention_mask=source_encoding['attention_mask'].to(device),
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=1.0,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    predictions = {
        qg_model.tokenizer.decode(
            generated_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    pred_str = ''.join(predictions)
    answer_start = pred_str.index("answer: ")
    question_start = pred_str.index("question: ")
    answer = pred_str[answer_start + 8: question_start - 1]
    question = pred_str[question_start + 10:]
    return {
        "answer": answer,
        "question": question
    }
