# -*- coding:utf-8 -*-
# @Time : 2022/7/28 17:01
# @Author: Jielong Tang
# @File : predict.py
import os

import paddle
from paddlenlp.transformers import ErnieForSequenceClassification,ErnieTinyTokenizer
from .dataset import label_vocab
from .utils import predict

def load_model(init_ckpt = None):
    paddle.set_device('gpu')
    tokenizer = ErnieTinyTokenizer.from_pretrained('ernie-tiny')
    model = ErnieForSequenceClassification.from_pretrained('ernie-tiny', num_classes=len(label_vocab))
    if not init_ckpt or not os.path.isfile(init_ckpt):
        raise Exception("init checkpoints {} not exist".format(init_ckpt))
    else:
        print('load model {}'.format(init_ckpt))
        state_dict = paddle.load(init_ckpt)
        model.set_dict(state_dict)
    return model,tokenizer

def do_predict(test_sample, model,tokenizer):
    data = dict(text=test_sample, label='NULL')
    id2label = {val: key for key, val in label_vocab.items()}
    result = predict(model, data, tokenizer, label_vocab, batch_size=1)
    return id2label[result[0]]

if __name__ == '__main__':
    '''
    下面为classifier标准用法，先导入模型load_model函数，然后do_predict进行inference。记得导入模型代码放在循环外。
    '''
    model,tokenizer = load_model(init_ckpt='../ckpt/role_judge_model/model_state.pdparams')
    print(do_predict(model = model,tokenizer=tokenizer, test_sample='里根号航母舰队'))


    # with open('./data/result_20220727_192612.txt','r',encoding='utf-8') as f:
    #     result_list = []
    #     for line in f:
    #         if line.strip().split('\t')[4] in ['发起者','承受者','使用器械']:
    #             test_sample = line.strip().split('\t')[6]
    #             result = do_predict(model = model,tokenizer=tokenizer, test_sample=test_sample)
    #             result_list.append(line.strip().split('\t')[4] + '\t'+test_sample + '\t'+ result)
    # with open('./data/result_show.txt','w',encoding='utf-8') as f:
    #     for line in result_list:
    #         f.write(line+'\n')


