# -*- coding:utf-8 -*-
# @Time : 2022/7/28 11:33
# @Author: Jielong Tang
# @File : train.py

import paddlehub as hub
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieForSequenceClassification,ErnieTinyTokenizer,LinearDecayWithWarmup
from .dataset import SelfDefinedDataset, label_vocab
from functools import partial
from .utils import  convert_example, create_dataloader,evaluate

batch_size = 128
max_seq_length = 42
learning_rate = 5e-5
epochs = 30
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数
weight_decay = 0.01

def train():
    paddle.set_device('gpu')

    train_ds = SelfDefinedDataset(data_path='data/train.xlsx', sheet_name="train_content")
    dev_ds = SelfDefinedDataset(data_path='data/train.xlsx', sheet_name="dev_content")

    model = ErnieForSequenceClassification.from_pretrained('ernie-tiny',num_classes=len(label_vocab))
    tokenizer = ErnieTinyTokenizer.from_pretrained('ernie-tiny')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): fn(list(map(trans_func, samples)))

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_function=trans_func)
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_function=trans_func)

    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 10 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (
                global_step, epoch, step, loss, acc))
            if global_step % 50 == 0:
                print("*************dev result****************")
                evaluate(model, criterion, metric, dev_data_loader)
                print("*************dev result****************")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        print("*************dev result****************")
        evaluate(model, criterion, metric, dev_data_loader)
        print("*************dev result****************")

    model.save_pretrained('./role_judge_model')
    tokenizer.save_pretrained('./role_judge_model')

if __name__ == '__main__':
    train()