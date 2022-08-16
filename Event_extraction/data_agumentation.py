# -*- coding:utf-8 -*-
# @Time : 2022/7/21 10:36
# @Author: Jielong Tang
# @File : data_agumentation.py
import json
import random
import pandas as pd
import numpy as np
import math
dict_role = {}
faqizhe = []
chenshouzhe = []
time = []
loc = []
tools = []
trigger = []
swap_list = ['时间','发起者','地点','承受者','使用器械']

def load_xlsx(data_path):
    df1 = pd.read_excel(data_path,header=0)
    for i in df1['发起者']:
        if pd.isna(i):
            continue
        else:
            faqizhe.append({'argument':i,'event_type':'-1'})
            faqizhe.append({'argument': i, 'event_type': '-1'})
    for i in df1['承受者']:
        if pd.isna(i):
            continue
        else:
            chenshouzhe.append({'argument':i,'event_type':'-1'})
            chenshouzhe.append({'argument': i, 'event_type': '-1'})
    for i in df1['地点']:
        if pd.isna(i):
            continue
        else:
            loc.append({'argument':i,'event_type':'-1'})
            loc.append({'argument': i, 'event_type': '-1'})
    for i in df1['使用器械']:
        if pd.isna(i):
            continue
        else:
            tools.append({'argument':i,'event_type':'-1'})
            tools.append({'argument': i, 'event_type': '-1'})


def radom_element(list_name):
    num_items = len(list_name)
    random_index = random.randrange(num_items)
    winner = list_name[random_index]
    return winner

def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            result.append(line.strip())
    return result

def showInfo(data_path):
    sentences = read_by_lines(data_path)  # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    for sent in sentences:
        if len(sent['event_list'])>0:
            for event in sent['event_list']:
                trigger.append({event['event_type']:event['trigger']})
                for arg in event['arguments']:
                    if arg['role'][4:] == '发起者':
                        faqizhe.append({'argument':arg['argument'], 'event_type':event['event_type'] })
                    elif arg['role'][4:] == '承受者':
                        chenshouzhe.append({'argument':arg['argument'], 'event_type':event['event_type'] })
                    elif arg['role'][4:] == '时间':
                        time.append({'argument':arg['argument'], 'event_type':event['event_type'] })
                    elif arg['role'][4:] == '地点':
                        loc.append({'argument':arg['argument'], 'event_type':event['event_type'] })
                    elif arg['role'][4:] == '使用器械':
                        tools.append({'argument':arg['argument'], 'event_type':event['event_type'] })

    print("**************")
    print("\n 发起者数量: {}".format(len(faqizhe)))
    print("**************")
    print("\n 承受者数量: {}".format(len(chenshouzhe)))
    print("**************")
    print("\n 时间数量: {}".format(len(time)))
    print("**************")
    print("\n 地点数量: {}".format(len(loc)))
    print("**************")
    print("\n 触发词数量: {}".format(len(trigger)))
    dict_role['时间'] = time
    dict_role['发起者'] = faqizhe
    dict_role['承受者'] = chenshouzhe
    dict_role['地点'] = loc
    dict_role['使用器械'] = tools

def extractElement(txts):
    #抽取事件主题
    txt_head = []
    for item in txts:
        assert len(item['text']) == len(item['labels'])
        starCur = 0
        tempCur = 0
        trigger = ''
        item_head = []
        while starCur < len(item['text']):
            if item['labels'][starCur] != 'O' and item['labels'][starCur].startswith('B-'):
                event = {}
                trigger = item['text'][starCur]
                tempCur = starCur
                starCur += 1
                while starCur < len(item['text']) and item['labels'][starCur].startswith('I-'):
                    trigger += item['text'][starCur]
                    starCur += 1
                event_type = item['labels'][tempCur][6:]
                event['argument'] = trigger
                event['type'] = event_type
                item_head.append(event)
            else:
                starCur += 1

        txt_head.append(item_head)
    return txt_head


def swapElement(t1,sentences):
    txts =[i for i in t1]
    txt_head = extractElement(txts)
    index = 0
    for txt in txt_head:
        if len(txt)>0:
            for element in txt:
                if element['type'] in swap_list:
                    start = sentences[index].find(element['argument'])
                    end = start+len(element['argument'])
                    new_ele = radom_element(dict_role[element['type']])
                    new_ele = new_ele['argument']
                    sentences[index] = sentences[index].replace(element['argument'],new_ele,1)
                    start_l = txts[index]['labels'][:start]
                    end_l = txts[index]['labels'][end:]
                    if len(element['argument'])>=len(new_ele):
                        medium_l = txts[index]['labels'][start:start+len(new_ele)]
                    else:
                        if txts[index]['labels'][start] != 'O':
                            add_l = 'I-' + txts[index]['labels'][start][2:]
                        else:
                            add_l = 'O'
                        medium_l = txts[index]['labels'][start:start+len(element['argument'])] + [add_l] * (len(new_ele) - len(element['argument']))
                    txts[index]['labels'] = start_l + medium_l + end_l
                    txts[index]['text'] = [i for i in list(sentences[index])]
                    # print('{} == {}'.format(len(txts[index]['labels']),len(txts[index]['text'])))
                    assert len(txts[index]['labels']) == len(txts[index]['text'])
        index +=1
    return txts,sentences

def read_tsv(data_path):
    sentences = []
    with open(data_path, 'r', encoding='utf-8') as fp:
        # skip the head line
        next(fp)
        txts = []
        for line in fp.readlines():
            words, labels = line.strip('\n').split('\t')
            words = words.split('\002')
            sentences.append(''.join(words))
            labels = labels.split('\002')
            # print('words {}   labels: {}'.format(words, labels))
            txts.append({'text': words, 'labels': labels})
            # word_ids.append(words)
            # label_ids.append(labels)
    return sentences,txts


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding="utf8") as outfile:
        outfile.write("text_a\tlabel\n")
        for d in data:
            out = "{}\t{}".format('\002'.join(d['text']),
                            '\002'.join(d['labels']))
            outfile.write(out + "\n")


if __name__ =="__main__":
    load_xlsx('../data/EE1.0/roles.xlsx')
    showInfo(data_path='../data/EE1.0/train.json')
    sentences, txts1 = read_tsv('../data/EE1.0/role/train.tsv')
    original = extractElement(txts1)
    print('***********原来元素\n')
    for i in original:
        print(i)
    print(len(txts1))
    txts1, sentences = swapElement(txts1, sentences)



    new_txt = extractElement(txts1)
    print('\n\n***********新元素\n')
    for i in new_txt:
        print(i)
    print(len(txts1))

    _, tw = read_tsv('../data/EE1.0/role/train.tsv')
    original = extractElement(tw)
    print('\n\n***********lao元素\n')
    for i in original:
        print(i)
    print(len(original))

    total = tw+txts1
    print(len(total))
    random.shuffle (total )
    write_by_lines(path='../data/EE1.0/role/train.tsv', data=total)

