
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
import NER_medNLP as ner

import codecs
from bs4 import BeautifulSoup
dict_key = {}

#%%
dict_key
# %%

training_data = './training_data/MedTxt-CR-JA-training-v2.xml'

def to_xml(data):
    __, __, __, key_attr = XtC.entities_from_xml(training_data, attrs=True)
    text = data['text']
    count = 0
    for i, entities in enumerate(data['entities_predicted']):
        if entities == "":
            return   
        span = entities['span']
        type_id = id_to_tags[entities['type_id']].split('_')
        tag = type_id[0]
        
        if not type_id[1] == "":
            attr = ' ' + value_to_key(type_id[1], key_attr) +  '=' + '"' + type_id[1] + '"'
        else:
            attr = ""
        
        add_tag = "<" + str(tag) + str(attr) + ">"
        text = text[:span[0]+count] + add_tag + text[span[0]+count:]
        count += len(add_tag)

        add_tag = "</" + str(tag) + ">"
        text = text[:span[1]+count] + add_tag + text[span[1]+count:]
        count += len(add_tag)
    return text

def entities_from_xml(file_name):#attrs=属性を考慮するか否か，考慮しないならFalse
    id_list = []
    title_list = []
    frequent_tags_attrs, __, __   = XtC.select_tags(attrs)
    #tags_value, dict_tags, id_to_tags = XtC.tags_parameter(frequent_tags_attrs)
    #dict_tags = dict(zip(frequent_tags_attrs, tags_value))#type_id への変換用
    with codecs.open(file_name, "r", "utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    for articles in soup.find_all("articles"):
        for elem in articles.find_all('article'):
            id_list.append(elem.attrs['id'])
            title_list.append(elem.attrs['title'])
    return id_list, title_list

def predict_entities(modelpath, sentences_list):
    model = ner.BertForTokenClassification_pl.load_from_checkpoint(
        checkpoint_path = modelpath + ".ckpt"
    ) 
    bert_tc = model.bert_tc.cuda()

    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = ner.NER_tokenizer_BIO.from_pretrained(
        MODEL_NAME,
        num_entity_type=len(frequent_tags_attrs) #Entityの数を変え忘れないように！
    )

    #entities_list = [] # 正解の固有表現を追加していく
    entities_predicted_list = [] # 抽出された固有表現を追加していく

    text_entities_set = []
    for dataset in sentences_list:
        text_entities = []
        for sample in tqdm(dataset):
            text = sample
            encoding, spans = tokenizer.encode_plus_untagged(
                text, return_tensors='pt'
            )
            encoding = { k: v.cuda() for k, v in encoding.items() } 
            
            with torch.no_grad():
                output = bert_tc(**encoding)
                scores = output.logits
                scores = scores[0].cpu().numpy().tolist()
            print(sum(scores))
                
            # 分類スコアを固有表現に変換する
            entities_predicted = tokenizer.convert_bert_output_to_entities(
                text, scores, spans
            )

            #entities_list.append(sample['entities'])
            entities_predicted_list.append(entities_predicted)
            text_entities.append({'text': text, 'entities_predicted': entities_predicted})
        text_entities_set.append(text_entities)
    return text_entities_set

def combine_sentences(text_entities_set, insert: str):
    documents = []
    for text_entities in tqdm(text_entities_set):
        document = []
        for t in text_entities:
            document.append(to_xml(t))
        documents.append('\n'.join(document))
    return documents

def add_id(documents, file_name):
    id_list, title_list = entities_from_xml(file_name) 
    doc_xml_list = []
    for i, d in enumerate(documents):
        doc_xml_list.append('<article id=' + '"'+ str(id_list[i])+ '"' +\
                ' title=' + '"' + str(title_list[i]) + '"' +'>\n')
        doc_xml_list.append(d)
        doc_xml_list.append('\n</article>\n')
    return doc_xml_list

# def to_zenkaku(text: str):
#     output = text.translate(str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)}))
#     return output

def value_to_key(value, key_attr):#attributeから属性名を取得
    global dict_key
    if dict_key.get(value) != None:
        return dict_key[value]
    for k in key_attr.keys():
        for v in key_attr[k]:
            if value == v:
                dict_key[v]=k
                return k

            
def add_metadata(filepath, content):       
    with codecs.open(filepath, "r", "utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    a_tag = soup.root#root内

    new_tag = soup.new_tag("articles")#articlesを新しいタグとして
    new_tag.string = '\n' + content #contentの中身をそのままnew_tagnのarticles内に入れ込む
    a_tag.articles.replace_with(new_tag) #articlesをnew_tagで置き換え
    xml = str(soup).replace('&lt;','<').replace('&gt;','>')
    return xml#タグを変換する


# %%
if __name__ == '__main__':
    filepath = './test_data/' + 'MedTxt-CR-JA-test.xml'
    attrs = True#属性考慮
    articles, articles_raw, entities, key_attr = XtC.entities_from_xml(filepath, attrs)#属性考慮するならTrue
    frequent_tags_attrs, _, id_to_tags = XtC.select_tags(attrs) #タグ取得

    __, __, id_to_tags = XtC.tags_parameter(frequent_tags_attrs) #idからタグへの変換

    #正規化されているテキスト
    dataset = XtC.create_dataset_no_tags(articles)
    #原文ママのテキスト
    dataset_r = XtC.create_dataset_no_tags(articles_raw)

    #固有表現予測
    text_entities_set = predict_entities('model_CR', dataset)
    text_entities_set_raw = []
    for i, texts_ent in enumerate(text_entities_set):
        sentence_raw = []
        for k, dic in enumerate(texts_ent):
            dic['text'] = dataset_r[i][k]
            sentence_raw.append(dic)
        text_entities_set_raw.append(sentence_raw)

    #ドキュメントにタグをつける
    documents = combine_sentences(text_entities_set_raw, '\n')

    doc_xml_list = "".join(add_id(documents, filepath))
    xml = add_metadata(filepath, doc_xml_list)


    #パス、ファイル名
    save_path_file = "MedTxt-CR-JA-test-Subtask1-baseline.xml"

    #ファイルの読み書き
    with open(save_path_file, "w") as f:
        f.write(str(xml))

# %%
