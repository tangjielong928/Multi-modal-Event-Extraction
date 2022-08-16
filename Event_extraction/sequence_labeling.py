# -*- coding:utf-8 -*-
# @Time : 2022/7/13 12:18
# @Author: Jielong Tang
# @File : sequence_labeling.py

"""
sequence labeling
"""
import time
import ast
import os
import json
import warnings
import random
import argparse
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer,RobertaForTokenClassification, RobertaTokenizer
from model import ptm_GRUCRF,FGM
from paddlenlp.metrics import ChunkEvaluator
from utils import read_by_lines, write_by_lines, load_dict

warnings.filterwarnings('ignore')

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--train_data", type=str, default=None, help="train data")
parser.add_argument("--dev_data", type=str, default=None, help="dev data")
parser.add_argument("--test_data", type=str, default=None, help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable.


def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(12,-1,-1):
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'encoder.layers.{layer}.' in n],
            'learning_rate': lr
        }
        parameters.append(layer_params)
        lr *= multiplier

    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'gru' in n or 'fc' in n
                   or 'pooling' in n],
        'learning_rate': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters

@paddle.no_grad()
def evaluate(model, criterion, metric, num_label, data_loader):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        loss = paddle.mean(
            criterion(logits.reshape([-1, num_label]), labels.reshape([-1])))
        losses.append(loss.numpy())
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds,
                                                     labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss

@paddle.no_grad()
def evaluate2(model, metric, data_loader):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        #crf loss
        preds = model(input_ids, seg_ids, lengths = seq_lens)
        # n_infer, n_label, n_correct = metric.compute(lens,preds,labels)
        n_infer, n_label, n_correct = metric.compute(seq_lens, preds,
                                                     labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss


def convert_example_to_feature(example,
                               tokenizer,
                               label_vocab=None,
                               max_seq_len=512,
                               no_entity_label="O",
                               ignore_label=-1,
                               is_test=False):
    tokens, labels = example
    tokenized_input = tokenizer(tokens,
                                return_length=True,
                                is_split_into_words=True,
                                max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len - 2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        return input_ids, token_type_ids, seq_len, encoded_label


class DuEventExtraction(paddle.io.Dataset):
    """DuEventExtraction"""

    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = ['[SEP]' if wd == '：'else wd for wd in words.split('\002')]
                print('words {}   labels: {}'.format(''.join(words), labels))
                labels = labels.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]


def do_train():
    paddle.set_device(args.device)
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    no_entity_label = "O"
    ignore_label = -1

    tokenizer = AutoTokenizer.from_pretrained("bert-wwm-ext-chinese")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    # model = AutoModelForTokenClassification.from_pretrained(
    #     "ernie-3.0-base-zh", num_classes=len(label_map))
    model = ptm_GRUCRF("bert-wwm-ext-chinese", num_class=len(label_map),gru_hidden_size=256)

    model = paddle.DataParallel(model)

    print("============start train==========")

    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    test_ds = DuEventExtraction(args.test_data, args.tag_path)

    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         label_vocab=train_ds.label_vocab,
                         max_seq_len=args.max_seq_len,
                         no_entity_label=no_entity_label,
                         ignore_label=ignore_label,
                         is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # token type ids
        Stack(dtype='int64'),  # sequence lens
        Pad(axis=0, pad_val=ignore_label, dtype='int64')  # labels
    ): fn(list(map(trans_func, samples)))

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    train_loader = paddle.io.DataLoader(dataset=train_ds,
                                        batch_sampler=batch_sampler,
                                        collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_size=args.batch_size,
                                      collate_fn=batchify_fn)
    test_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_size=args.batch_size,
                                       collate_fn=batchify_fn)

    num_training_steps = len(train_loader) * args.num_epoch
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    # assert  len(model.named_parameters()) == len(model.parameters())
    for n,p in model.named_parameters():
        print(n)

    # parameter = [{'params': [para for name,para in model.named_parameters() if 'crf' not in name], 'learning_rate': args.learning_rate},
    #  {'params': [para for name,para in model.named_parameters() if 'crf' in name], 'learning_rate': args.learning_rate*20}]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=2e-5,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)


    metric = ChunkEvaluator(label_list=train_ds.label_vocab.keys(),
                            suffix=False)
    criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    fgm = FGM(model=model)
    now = str(int(time.time()))
    step, best_f1 = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, seq_lens,
                  labels) in enumerate(train_loader):
            logits = model(input_ids,
                           token_type_ids,seq_lens,labels).reshape([-1, train_ds.label_num])
            loss = paddle.mean(criterion(logits, labels.reshape([-1])))
            # loss = paddle.mean(logits)
            loss.backward()
            fgm.attack()
            logits_adv = model(input_ids,
                           token_type_ids,seq_lens,labels).reshape([-1, train_ds.label_num])
            loss_adv = paddle.mean(criterion(logits_adv, labels.reshape([-1])))
            loss_adv.backward()
            fgm.restore()
            optimizer.step()
            # optimizer2.clear_grad()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(
                    f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}'
                )
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                p, r, f1, avg_loss = evaluate(model, criterion, metric,
                                              len(label_map), dev_loader)
                # p, r, f1, avg_loss = evaluate2(model, metric,
                #                                dev_loader)

                print(f'dev step: {step} - loss: {loss_item:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                        f'f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    best_f1 = f1
                    print(f'==============================================save best model ' \
                            f'best performerence {best_f1:5f}')
                    paddle.save(model.state_dict(),
                                '{}/BertGRU{}.pdparams'.format(args.checkpoints,now))
            step += 1

    # save the final model
    # if rank == 0:
    #     paddle.save(model.state_dict(),
    #                 '{}/final.pdparams'.format(args.checkpoints))


def do_predict():
    paddle.set_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    # model = AutoModelForTokenClassification.from_pretrained(
    #     "ernie-3.0-base-zh", num_classes=len(label_map))
    model = ptm_GRUCRF("ernie-3.0-base-zh", num_class=len(label_map), gru_hidden_size=256)

    no_entity_label = "O"
    ignore_label = len(label_map)

    print("============start predict==========")
    # if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
    #     raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    # else:
    #     state_dict = paddle.load(args.init_ckpt)
    #     model.set_dict(state_dict)
    #     print("Loaded parameters from %s" % args.init_ckpt)

    state_dict = paddle.load('../ckpt/EE1.0/role/bestGRU.pdparams')
    model.set_dict(state_dict)

    if args.tag_path.split('/')[-1] == 'role_tag.dict':
        if not os.path.isfile('../ckpt/EE1.0/trigger/test_pred.json'):
            raise Exception("Trigger prediction result not exists, please check and infer the trigger first!")
        else:
            sentences = read_by_lines(args.predict_data)  # origin data format
            sentences = [json.loads(sent) for sent in sentences]
            txts1 = read_by_lines('../ckpt/EE1.0/trigger/test_pred.json')
            txts = [json.loads(sent) for sent in txts1]
            txt_head = []
            for item in txts:
                assert len(item['text']) == len(item['labels'])
                starCur = 0
                tempCur = 0
                trigger = ''
                item_head = []
                while starCur<len(item['text']):
                    if item['labels'][starCur] != 'O' and item['labels'][starCur].startswith('B-'):
                        event = {}
                        trigger = item['text'][starCur]
                        tempCur = starCur
                        starCur += 1
                        while starCur<len(item['text']) and item['labels'][starCur].startswith('I-'):
                            trigger += item['text'][starCur]
                            starCur += 1
                        event_type = item['labels'][tempCur][2:]
                        event['trigger'] = trigger
                        event['event_type'] = event_type
                        item_head.append(event)
                    else:
                        starCur+=1

                txt_head.append(item_head)

            result = []
            for i in range(len(sentences)):
                if len(txt_head[i]) > 0:
                    for txts in txt_head[i]:
                                result.append({'id':sentences[i]['id'] , 'text': txts['event_type'] #+txts['trigger']
                                                                                 +'：'+sentences[i]['text']})
                # else:
                #     result.append({'id': sentences[i]['id'], 'text': sentences[i]['text']})

            for i in result:
                print(i)
            sentences = result
    else:
        # load data from predict file
        sentences = read_by_lines(args.predict_data)  # origin data format
        sentences = [json.loads(sent) for sent in sentences]

    # sentences = read_by_lines(args.predict_data)  # origin data format
    # sentences = [json.loads(sent) for sent in sentences]

    encoded_inputs_list = []
    for sent in sentences:
        sent = sent["text"].replace(" ", "\002")
        input_ids, token_type_ids, seq_len = convert_example_to_feature(
            [list(sent), []],
            tokenizer,
            max_seq_len=args.max_seq_len,
            is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # token_type_ids
        Stack(dtype='int64')  # sequence lens
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [
        encoded_inputs_list[i:i + args.batch_size]
        for i in range(0, len(encoded_inputs_list), args.batch_size)
    ]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        seq_lens = paddle.to_tensor(seq_lens)
        preds = model(input_ids, token_type_ids,seq_lens)
        print(preds)
        probs = [pred for pred in preds.numpy()]
        # probs = F.softmax(logits, axis=-1)
        # probs_ids = np.argmax(probs, -1)
        for p_list, p_ids, seq_len in zip(probs, probs,
                                          seq_lens.numpy().tolist()):
            # prob_one = [
            #     p_list[index][pid]
            #     for index, pid in enumerate(p_ids[1:seq_len - 1])
            # ]
            label_one = [id2label[pid] for pid in p_ids[1:seq_len - 1]]
            results.append({"probs": [], "labels": label_one})
    assert len(results) == len(sentences)
    print_result = []
    for sent, ret in zip(sentences, results):
        row = ''
        t = ''
        for i in range(len(ret['labels'])):
            if ret['labels'][i] != 'O':
                row += (ret['labels'][i])
                t+=(sent['text'][i])
        print_result.append({'id': sent['id'],'text': t, 'labels': row})
    # sentences = [json.dumps(sent, ensure_ascii=False) for sent in results]
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in print_result]
    now = int(time.time())
    predict_save_path = './role_judge_model/EE1.0/role/test_pred{}.json'.format(now)
    write_by_lines(predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), predict_save_path))


if __name__ == '__main__':

    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()