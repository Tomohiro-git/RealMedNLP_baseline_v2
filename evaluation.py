# %%
import from_XML_to_json as XtC
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
# 8-19
def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0 # 固有表現(正解)の個数
    num_predictions = 0 # BERTにより予測された固有表現の個数
    num_correct = 0 # BERTにより予測のうち正解であった固有表現の数

    # それぞれの文章で予測と正解を比較。
    # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。
    for entities, entities_predicted \
        in zip(entities_list, entities_predicted_list):

        if type_id:
            entities = [ e for e in entities if e['type_id'] == type_id ]
            entities_predicted = [ 
                e for e in entities_predicted if e['type_id'] == type_id
            ]
            
        get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set( get_span_type(e) for e in entities )
        set_entities_predicted = \
        set_entities_predicted = \
            set( get_span_type(e) for e in entities_predicted )

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len( set_entities & set_entities_predicted )

    # 指標を計算
    precision = None
    recall = None
    f_value = None
    if num_predictions != 0:
        precision = num_correct/num_predictions # 適合率
    if num_entities != 0:
        recall = num_correct/num_entities # 再現率
    if precision != None and recall != None and precision + recall != 0:
        f_value = 2*precision*recall/(precision+recall) # F値

    result = {
            'num_entities': num_entities,
            'num_predictions': num_predictions,
            'num_correct': num_correct,
            'precision': precision,
            'recall': recall,
            'f_value': f_value
    }

    return result

# %%

result = []

model = BertForTokenClassification_pl.load_from_checkpoint(
    checkpoint_path="no_validation"+".ckpt"
) 
bert_tc = model.bert_tc.cuda()

entities_list = [] # 正解の固有表現を追加していく
entities_predicted_list = [] # 抽出された固有表現を追加していく
text_entities = []
for sample in tqdm(dataset_test):
    text = sample['text']
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    )
    encoding = { k: v.cuda() for k, v in encoding.items() } 
    
    with torch.no_grad():
        output = bert_tc(**encoding)
        scores = output.logits
        scores = scores[0].cpu().numpy().tolist()
        
    # 分類スコアを固有表現に変換する
    entities_predicted = tokenizer.convert_bert_output_to_entities(
        text, scores, spans
    )

    entities_list.append(sample['entities'])
    entities_predicted_list.append(entities_predicted)
    text_entities.append({'text': text, 'entities': sample['entities'], 'entities_predicted': entities_predicted})




print(evaluate_model(entities_list, entities_predicted_list))
    #result.append(evaluate_model(entities_list, entities_predicted_list))


# %%
precision = [r['precision'] for r in result]
recall = [r['recall'] for r in result]
f_value = [r['f_value'] for r in result]

# %%
print('f_value', ['{:.3f}'.format(f) for f in f_value])
print('precision:{:.3f}'.format(np.average(precision)))
print('recall:{:.3f}'.format(np.average(recall)))
print('f_value:{:.3f}'.format(np.average(f_value)))


# %%
#各タグごとの評価
def evaluation_testset(filepath):
    model = BertForTokenClassification_pl.load_from_checkpoint(
        checkpoint_path=filepath+".ckpt"
    ) 
    bert_tc = model.bert_tc.cuda()

    entities_list = [] # 正解の固有表現を追加していく
    entities_predicted_list = [] # 抽出された固有表現を追加していく
    text_entities = []
    for sample in tqdm(dataset_test):
        text = sample['text']
        encoding, spans = tokenizer.encode_plus_untagged(
            text, return_tensors='pt'
        )
        encoding = { k: v.cuda() for k, v in encoding.items() } 
        
        with torch.no_grad():
            output = bert_tc(**encoding)
            scores = output.logits
            scores = scores[0].cpu().numpy().tolist()
            
        # 分類スコアを固有表現に変換する
        entities_predicted = tokenizer.convert_bert_output_to_entities(
            text, scores, spans
        )

        entities_list.append(sample['entities'])
        entities_predicted_list.append(entities_predicted)
        text_entities.append({'text': text, 'entities': sample['entities'], 'entities_predicted': entities_predicted})
        
        evaluate =[]
        for i in range(1, len(frequent_tags_attrs)+1):
            evaluate.append(evaluate_model(entities_list, entities_predicted_list, type_id=i))
    
    df_eval = pd.DataFrame(evaluate)
    tags_value = [int(i) for i in range(1, len(frequent_tags_attrs)+1)]
    dict_tags = dict(zip(frequent_tags_attrs, tags_value))#type_id への変換用
    df_eval.insert(0, 'type', dict_tags)
    return df_eval[df_eval['num_entities']!=0]

# %%
# データフレームのリストを作成
results = []

evaluation_testset("no_validation")
dict_tag