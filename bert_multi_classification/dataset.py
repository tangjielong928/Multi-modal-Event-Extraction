import sys

sys.path.append('../..')
import torch
from torch.utils.data import Dataset
# 这里要显示的引入BertFeature，不然会报错
from data_preprocess import get_out, Processor
import config


class MLDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).float() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
        }

        data['labels'] = self.labels[index]

        return data


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.log_dir = 'logs/'
    args.max_seq_len = 128
    args.bert_dir = 'model/bert-base-chinese/'

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('../data/MutiClass/processed_data/labels.txt', 'r', encoding='utf-8') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[id] = label
    print(label2id)

    train_out = get_out(processor, '../data/MutiClass/processed_data/train.json', args, label2id, 'train')
    features, callback_info = train_out
    train_dataset = MLDataset(features)
    for data in train_dataset:
        print(data['token_ids'])
        print(data['attention_masks'])
        print(data['token_type_ids'])
        print(data['labels'])
        break
