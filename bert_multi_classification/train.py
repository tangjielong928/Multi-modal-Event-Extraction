import os
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import config
import data_preprocess
import dataset
import models
import utils.utils as ut


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.models_4 = models.BertForMultiLabelClassification(args)
        self.optimizer = torch.optim.Adam(params=self.models_4.parameters(), lr=self.args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.models_4.to(self.device)

    def load_ckp(self, models_4, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        models_4.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return models_4, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)


    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 100
        best_dev_accuracy = 0.0
        for epoch in range(args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.models_4.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.models_4(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                    logger.info(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy,
                                                                                                   micro_f1, macro_f1))
                    if accuracy > best_dev_accuracy:
                        logger.info("------------>保存当前最好的模型")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.models_4.state_dict(),
                            # 'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_accuracy = accuracy
                        checkpoint_path = os.path.join(self.args.output_dir, 'best3.pt')
                        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self):
        self.models_4.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.models_4(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.6).astype(int)
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        models_4 = self.models_4
        optimizer = self.optimizer
        models_4, epoch, loss = self.load_ckp(models_4,checkpoint_path)
        models_4.eval()
        models_4.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = models_4(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.6).astype(int)
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args):
        models_4 = self.models_4
        optimizer = self.optimizer
        checkpoint = os.path.join(args.output_dir, 'best3.pt')
        models_4, optimizer, epoch, loss = self.load_ckp(models_4, optimizer, checkpoint)
        models_4.eval()
        models_4.to(self.device)
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
            outputs = models_4(token_ids, attention_masks, token_type_ids)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            outputs = (np.array(outputs) > 0.5).astype(int)
            outputs = np.where(outputs[0] == 1)[0].tolist()
            if len(outputs) != 0:
                outputs = [id2label[i] for i in outputs]
                return outputs
            else:
                return '无事件'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        # confusion_matrix = multilabel_confusion_matrix(targets, outputs)
        report = classification_report(targets, outputs, target_names=labels)
        return report


if __name__ == '__main__':
    args = config.Args().get_parser()
    ut.set_seed(args.seed)
    ut.set_logger(os.path.join(args.log_dir, 'main.log'))

    processor = data_preprocess.Processor()

    label2id = {}
    id2label = {}
    with open('../data/MutiClass/processed_data/labels.txt', 'r', encoding='utf-8') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    print(label2id)

    train_out = data_preprocess.get_out(processor, '../data/MutiClass/processed_data/train.json', args, label2id, 'train')
    train_features, train_callback_info = train_out
    train_dataset = dataset.MLDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)

    dev_out = data_preprocess.get_out(processor, '../data/MutiClass/processed_data/dev.json', args, label2id, 'dev')
    dev_features, dev_callback_info = dev_out
    dev_dataset = dataset.MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    trainer = Trainer(args, train_loader, dev_loader, dev_loader)
    # 训练
    trainer.train()

    # 测试
    logger.info('========进行测试========')
    checkpoint_path = '../ckpt/MultiClass/best3.pt'
    total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
    logger.info(
        "【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy, micro_f1,
                                                                                    macro_f1))
    report = trainer.get_classification_report(test_outputs, test_targets, labels)
    logger.info(report)