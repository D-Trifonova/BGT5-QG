from typing import List, Dict
import tqdm.notebook as tq
from tqdm.notebook import tqdm
import json
import pandas as pd
import numpy as np

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    MT5ForConditionalGeneration as MT5ForConditionalGeneration,
    #T5Tokenizer,
    MT5TokenizerFast as MT5TokenizerFast
)

pl.seed_everything(42)

SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 128

SEP_TOKEN = '<sep>'

MODEL_NAME = "qg_model"
#MODEL_NAME = "ag_chatgpt"

tokenizer = MT5TokenizerFast.from_pretrained(MODEL_NAME)
print('tokenizer len: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)

class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True, device_map = 'cuda') # auto
        self.model.resize_token_embeddings(TOKENIZER_LEN)  # resizing after adding new tokens to the tokenizer

def generate(qgmodel: QGModel, answer: str, context: str) -> str:
    source_encoding = tokenizer(
        'answer: {} context: {}'.format(answer, context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'].to("cuda:0"),
        attention_mask=source_encoding['attention_mask'].to("cuda:0"),
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=1.0,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(preds)


#model = QGModel.load_from_checkpoint('checkpoints_mt5/best-checkpoint-v4.ckpt')
model = QGModel()

context = "Калкулирането на себестойността е важен аспект на управленското счетоводство и има за цел да определи разходите, свързани с производството на стоки или услугите, предлагани от компанията. В управленското счетоводство, се използват различни методи и системи, например, фактическо калкулиране или нормативно калкулиране. Методите за калкулиране помагат на ръководството да взема информирани решения относно ценовата политика и разходите. Често, предприятията разчитат на данни от пълна себестойност или непълна себестойност, за да оценят производствените си разходи. Тези системи са част от структурите на управление, които имат за цел да оптимизират производствения процес и да повишат ефективността на бизнеса."

#answer = "Да определи производствени разходи."
#answer = "ръководството да взема информирани решения"
answer = "структури на управление"
#answer = "[MASK]"
#answer = "производство на стоки или услугите, предлагани от компанията"
#answer = "управление"

#context = "Симеон Монов е най-добрият програмист!"
#answer = "Симеон Монов"

context = "Източници на привлечен капитал са различните кредитори на предприятието. Участието на техните средства във финансовия оборот на предприятието се нарича задължения към кредиторите. Предприятието привлича и ползва в оборота си средства на кредиторите си като купува материали, стоки и други активи, ползва услуги, за които не се разплаща веднага при получаването им, а в рамките на определен срок, или като задържа изплащането на начислените за негова сметка заплати, удръжки от заплатите, данъци, осигуровки и други задължения при осъществяването на дейността до настъпване на определения за целта срок."
answer = "средства на кредиторите"

generated = generate(model, answer, context) 

print(generated)
