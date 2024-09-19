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

from pytorch_lightning.loggers import WandbLogger
from ag_datamodule import AGDataModule

pl.seed_everything(42)

SOURCE_MAX_TOKEN_LEN = 384
TARGET_MAX_TOKEN_LEN = 128

SEP_TOKEN = '<sep>'

MODEL_NAME = "ag_model"

tokenizer = MT5TokenizerFast.from_pretrained(MODEL_NAME)
print('tokenizer len: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)

class AGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(TOKENIZER_LEN)  # resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)

def generate(agmodel: AGModel, context: str) -> str:
    source_encoding = tokenizer(
        'context: {}'.format(context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = agmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
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


#model = AGModel.load_from_checkpoint('checkpoints_mt5/best-checkpoint-v4.ckpt')
model = AGModel()

context = "Калкулирането на себестойността е важен аспект на управленското счетоводство и има за цел да определи разходите, свързани с производството на стоки или услугите, предлагани от компанията. В управленското счетоводство, се използват различни методи и системи, например, фактическо калкулиране или нормативно калкулиране. Методите за калкулиране помагат на ръководството да взема информирани решения относно ценовата политика и разходите. Често, предприятията разчитат на данни от пълна себестойност или непълна себестойност, за да оценят производствените си разходи. Тези системи са част от структурите на управление, които имат за цел да оптимизират производствения процес и да повишат ефективността на бизнеса."

#context = context[:200]

context = "Симеон Монов е най-добрият програмист!"
#answer = "най-добрият програмист"
generated = generate(model, context) 

print(generated)
