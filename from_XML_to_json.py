
# %%
import codecs
from bs4 import BeautifulSoup
import unicodedata
import re
import pandas as pd
import json
import argparse
import sys

#%%
def select_tags(attrs = True):#attributes
    if attrs:
        frequent_tags_attrs= ['d_', 'd_positive', 'd_suspicious', 'd_negative', 'd_general', 'a_', 'f_', 'c_',\
                     'timex3_', 'timex3_date', 'timex3_time', 'timex3_duration', 'timex3_set', 'timex3_age', 'timex3_med', 'timex3_misc',\
                     't-test_', 't-test_executed', 't-test_negated', 't-test_scheduled', 't-test_other','t-key_', 't-key_scheduled', 't-val_',\
                     'm-key_executed', 'm-key_negated', 'm-key_scheduled', 'm-key_other',\
                     'm-val_', 'm-val_negated', 'm-val_executed', 'm-val_scheduled',\
                     'r_scheduled', 'r_executed', 'r_negated', 'r_other',\
                     'cc_scheduled', 'cc_executed', 'cc_negated', 'cc_other']
    #else:
        #frequent_tags_attrs = ['d_', 'a_', 'f_', 'timex3_', 't-test_', 't-key_', 't-val_', 'm-key_', 'm-val_', 'r_', 'cc_', 'c_'] 
    frequent_tags= ['d', 'a', 'f', 'timex3', 't-test', 't-key', 't-val', 'm-key', 'm-val', 'r', 'cc']   
    attributes_keys = ['type', 'certainty', 'state']
    return frequent_tags_attrs, frequent_tags, attributes_keys


def tags_parameter(frequent_tags_attrs):
    tags_value = [int(i) for i in range(1, len(frequent_tags_attrs)+1)]
    dict_tags = dict(zip(frequent_tags_attrs, tags_value))#type_idへの変換用
    id_to_tags = {v: k for k, v in dict_tags.items()}#id_to_typeへの変換用
    return tags_value, dict_tags, id_to_tags

def entities_from_xml(file_name, attrs = True):#attrs=属性を考慮するか否か，考慮しないならFalse
    frequent_tags_attrs, frequent_tags, attributes_keys = select_tags(attrs = True)
    tags_value, dict_tags, __ = tags_parameter(frequent_tags_attrs)
    with codecs.open(file_name, "r", "utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    key_type = set()
    key_certainty = set()
    key_state = set()
    

    for elem_articles in soup.find_all("articles"):#articles内のarticleを一つずつ取り出す
        entities = []
        articles = []
        articles_raw = []
        for elem in elem_articles.find_all('article'):#article内の要素を一つずつ取り出す
            entities_article = []
            text_list = []
            text_raw_list = []
            pos1 = 0
            pos2 = 0
            for child in elem:#取り出した要素に対して，一つずつ処理する
                #（タグのないものについても要素として取得されるので，位置(pos)はずれない）                
                text = unicodedata.normalize('NFKC', child.string)#正規化
                text_raw = child.string#正規化する前のテキスト保存
                #text = text.replace('。', '.')#句点を'.'に統一, sentenceの分割に使うため．
                pos2 += len(text)#終了位置を記憶
                if child.name in frequent_tags:#特定のタグについて，固有表現の表現形，位置，タグを取得
                    attr = ""#属性を入れるため
                    if 'type' in child.attrs:#typeがある場合には
                        attr = child.attrs['type']
                        key_type.add(attr)
                    if 'certainty' in child.attrs:#certaintyがある場合には
                        attr = child.attrs['certainty']
                        key_certainty.add(attr)
                    if 'state' in child.attrs:#stateがある場合には
                        attr = child.attrs['state']
                        key_state.add(attr)
                    if not attrs:#attrs=属性を考慮するか否か，考慮しないならFalse
                        attr = ""
                    entities_article.append({'name':text, 'span':[pos1, pos2],\
                        'type_id':dict_tags[str(child.name)+'_'+str(attr)],\
                        'type':str(child.name)+'_'+str(attr)})
                pos1 = pos2#次のentityの開始位置を設定
                text_list.append(text)
                text_raw_list.append(text_raw)
            articles.append("".join(text_list))
            articles_raw.append("".join(text_raw_list))
            entities.append(entities_article)
    key_attr = {'type':key_type, 'certainty':key_certainty, 'state':key_state}
    return articles, articles_raw, entities, key_attr

def to_sentences(articles):  #articleをsentenceにばらす
    sentences = []
    for s in articles:
        sentences.append(re.split(r'\n', s))
    return sentences

def sentence_with_NE(sentences, entities):
    # 文単位にばらしたものに，エンティティを付与し直す
    texts_dataset = []
    for i in range(len(sentences)):
        pos = 0
        text_dataset = []
        for k in range(len(sentences[i])):
            text_dataset.append({'ID': (i, k)})#IDを追加してみる，元に戻す時用, [何番目のarticleか，何番目のsentenceか]
            sentence = sentences[i][k].replace('\\n', "")#冒頭が\\nになっていてもしかしたらずれの原因になっているかも
            text_dataset[k].update({'text': sentence})#テキスト追加
            tmp_entities = []
            while entities[i][0]['span'][1] <= len(sentence) + pos:#終了位置が超えていたら，次の文へ
                entities[i][0]['span'] = [entities[i][0]['span'][0] - pos,\
                    entities[i][0]['span'][1] - pos]#span入力
                tmp_entities.append(entities[i][0])
                del entities[i][0]#entity入れたら消していく
                if not entities[i]:#entitiyがなくなったら終わり
                    break
            text_dataset[k].update({'entities': tmp_entities})        
            if not entities[i]:#entitiyがなくなったら終わり
                break
            pos += len(sentence)+1#'\n.'でスプリットしているので+1
        texts_dataset.append(text_dataset)
        
    dataset_t = []
    for i in texts_dataset:#ネストをはずす
        dataset_t.extend(i)
        
    dataset = []
    for d in dataset_t:
        if d['text']!="":
            dataset.append(d)    
    return dataset

def create_dataset_no_tags(articles):#タグがないxmlファイルからテキストをセンテンスごとにリスト化する．
    sentences = to_sentences(articles)
        
    dataset = [] 
    for sentence in sentences:
        dataset_tmp = []
        for d in sentence:
            if d!="":
                dataset_tmp.append(d)
        dataset.append(dataset_tmp)
    return dataset



# %%
if __name__ == '__main__':
    input_file = './training_data/' + 'MedTxt-RR-JA-training.xml'
    articles, __, entities, __ = entities_from_xml(input_file, attrs=True)#属性考慮するならTrue
    sentences = to_sentences(articles)
    dataset_t = sentence_with_NE(sentences, entities)

    json_file = './training_data/' + 'subtask1-RR-text_with_span.json'
    with open(json_file, 'w') as f:
        json.dump(dataset_t, f, ensure_ascii=False)



# %%

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_file', nargs='?', type=argparse.FileType("r"),
            default=sys.stdin, help="input file path")
    parser.add_argument("-o", '--output_file', nargs='?', type=argparse.FileType("w"),
            default=sys.stdout, help="output file path")
    parser.add_argument("-m", '--model_name', default='BERT', help="model file directory")
    parser.add_argument("-n", '--normalizer', default='dict', help="dictionary file directory")
    parser.add_argument("-f", '--output_format', default='xml', help="output format (xml or dict). default is xml")
    args = parser.parse_args()