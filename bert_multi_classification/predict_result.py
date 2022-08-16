# Author:xyy
# CreatTime:2022-07-11
# FileName:predict_result
# Description:


import os
import logging
import torch
import json
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
import config
import data_preprocess
import models
import utils.utils as ut
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = models.BertForMultiLabelClassification(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, epoch, loss

    def test(self, checkpoint_path):
        model = self.model
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        model = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.6).astype(int)
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args):
        model = self.model
        optimizer = self.optimizer
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model, epoch, loss = self.load_ckp(model, checkpoint)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.max_seq_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)

            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            outputs = (np.array(outputs) > 0.5).astype(int)
            outputs = np.where(outputs[0] == 1)[0].tolist()
            if len(outputs) != 0:
                outputs = [id2label[i] for i in outputs]
                return outputs
            else:
                return ["无事件"]


if __name__ == '__main__':
    args = config.Args().get_parser()
    ut.set_seed(args.seed)
    ut.set_logger(os.path.join(args.log_dir, 'final_main.log'))

    processor = data_preprocess.Processor()

    label2id = {}
    id2label = {}
    with open('../data/MutiClass/processed_data/labels.txt', 'r', encoding='utf-8') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    # 预测
    trainer = Trainer(args, None, None, None)
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    res_list = []
    # 读取test1.json里面的数据
    with open(os.path.join('../data/raw_data/test/test_no_ann.json'), 'r', encoding='utf-8')as fp:
        lines = json.load(fp)
        # lines = fp.read().strip().split('\n')

        print("****************Now start our multi-labels inference!!*************")
        for line in tqdm(lines):
            id = line["id"]
            text = line["my_text"]
            result = trainer.predict(tokenizer, text, id2label, args)
            # print(result)
            res_list.append({"id": id, "text": text, "class": result})

    with open('../data/result/multi_label/result.json', 'w', encoding='utf-8') as f:
        # f.write("[\n")
        # for item in res_list:
        #     f.write(str(item) + ",\n")
        # f.write("]")
        f.write(json.dumps(res_list,ensure_ascii=False))
