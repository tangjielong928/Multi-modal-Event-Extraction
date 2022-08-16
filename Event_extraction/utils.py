# -*- coding:utf-8 -*-
# @Time : 2022/7/13 12:17
# @Author: Jielong Tang
# @File : utils.py


import hashlib
import json

def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()


def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding="utf8") as outfile:
        [outfile.write(d + "\n") for d in data]


def text_to_sents(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    paragraphs = text.split("\n")
    ret = []
    for para in paragraphs:
        if para == u"":
            continue
        sents = [u""]
        for s in para:
            sents[-1] += s
            if s in deliniter_symbols:
                sents.append(u"")
        if sents[-1] == u"":
            sents = sents[:-1]
        ret.extend(sents)
    return ret


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    if len(text) != len(labels):
        # 韩文回导致label 比 text要长
        labels = labels[:len(text)]
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def val_generate(data_path,pre_result):
    # 验证集生成从多标签分类结果
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = data  # origin data format
    result = []
    set_list = []
    # for sent in sentences:
    #     for clas in sent['class']:
    #         if clas != '无事件':
    #             item = {}
    #             item['id'] = sent['id']
    #             item['text'] = clas+'：'+ sent['text'].rstrip()
    #             set_list.append(item)
    for sent in sentences:
        for clas in sent['class']:
            if clas != '无事件':
                set_list.append((sent['id'],clas))
    print(set_list)
    for txt in pre_result:
        a = (txt['id'],txt['event_type'])
        if a in set_list:
            result.append(txt)
    return result

def wgm_trans_decodes(ds, decodes, lens, label_vocab):
    #将decodes和lens由列表转换为数组
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    #先使用zip形成元祖（编号, 标签），然后使用dict形成字典
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))
    #保存所有句子解析结果的列表
    results=[]
    #初始化编号
    inNum = 1;
    #逐个处理待转换的标签编码列表
    for idx, end in enumerate(lens):
        #句子单字构成的数组
        sent_array = ds.data[idx][0][:end]
        #句子单字标签构成的数组
        tags_array = [id_label[x] for x in decodes[idx][1:end]]
        #初始化句子和解析结果
        sent = "";
        tags = "";
        #将字符串数组转换为单个字符串
        for i in range(end-2):
            #pdb.set_trace()
            #单字直接连接，形成句子
            sent = sent + sent_array[i]
            #标签以空格连接
            if i > 0:
                tags = tags + " " + tags_array[i]
            else:#第1个标签
                tags = tags_array[i]
        #构成结果串：编号+句子+标签序列，中间用“\u0001”连接
        current_pred = str(inNum) + '\u0001' + sent + '\u0001' + tags + "\n"
        #pdb.set_trace()
        #添加到句子解析结果的列表
        results.append(current_pred)
        inNum = inNum + 1
    return results


if __name__ == "__main__":
    # s = "xxdedewd"
    # print(cal_md5(s.encode("utf-8")))
    print(load_dict('../data/EE1.0/trigger_tag.dict'))