
# %%
import from_XML_to_json as XtC
import NER_medNLP as ner
import itertools
import random
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import unicodedata

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForTokenClassification
import pytorch_lightning as pl




# %%
# データセットの分割(8:2に分割)
def load_json(filepath):
    tf = open(filepath, "r")
    dataset = json.load(tf)

    n = len(dataset)
    n_train = int(n*0.8)
    dataset_train = dataset
    dataset_val= dataset[n_train:]
    return dataset_train, dataset_val

# %%
def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    dataset_for_loader = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        print(encoding)
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)
    return dataset_for_loader

# %%
def training(model_name: str, dataset_train, dataset_val):
    # トークナイザのロード
    # 日本語学習済みモデル
    # 日本語学習済みモデル
    frequent_tags_attrs, _, _ = XtC.select_tags(attrs=True) #タグ取得  
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    # 固有表現のカテゴリーの数`num_entity_type`を入力に入れる必要がある。
    tokenizer = ner.NER_tokenizer_BIO.from_pretrained(
        MODEL_NAME,
        num_entity_type=len(frequent_tags_attrs) #Entityの数を変え忘れないように！
    )

    # データセットの作成
    max_length = 128
    dataset_train_for_loader = create_dataset(
        tokenizer, dataset_train, max_length
    )
    dataset_val_for_loader = create_dataset(
        tokenizer, dataset_val, max_length
    )

    # データローダの作成
    dataloader_train = DataLoader(
        dataset_train_for_loader, batch_size=32, shuffle=True
    )
    dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)

    # ファインチューニング
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='model_BIO/'
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=20,
        callbacks=[checkpoint]
    )

    # PyTorch Lightningのモデルのロード
    num_entity_type = len(frequent_tags_attrs)#entityの数を変え忘れないように！！
    num_labels = 2*num_entity_type+1
    model = ner.BertForTokenClassification_pl(
        MODEL_NAME, num_labels=num_labels, lr=1e-5
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    trainer.save_checkpoint(model_name+".ckpt")
    return

# %%
if __name__ == '__main__':
    filepath = './training_data/' + 'subtask1-RR-text_with_span.json'
    dataset_train, dataset_val = load_json(filepath)
    training('model_RR', dataset_train, dataset_val)

# %%

# %%
