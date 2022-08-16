# -*- coding:utf-8 -*-
# @Time : 2022/7/28 13:40
# @Author: Jielong Tang
# @File : dataset.py
import paddle
import paddlenlp
import pandas as pd

label_vocab = {'airplane': 0, 'boat': 1,'missile':2,'other':3,'submarine':4,'truck':5}


class SelfDefinedDataset(paddle.io.Dataset):
    def __init__(self, data_path,sheet_name):
        super(SelfDefinedDataset, self).__init__()
        self.data =[]
        label_vocab = {'airplane': 0, 'boat': 1, 'missile': 2, 'other': 3, 'submarine': 4, 'truck': 5}
        train_df = pd.read_excel(data_path, header=0, sheet_name=sheet_name)
        for txt, label in zip(train_df['Column6'].tolist(), train_df['Column7'].tolist()):
            if len(txt)>0 and len(label)>0:
                self.data.append(dict(text=txt, label=label_vocab[label]))


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return ['0','1','2','3','4','5',]



def xlsx_to_list(data_path):
    res_list = []
    dedv_list = []
    train_df = pd.read_excel(data_path, header=0,sheet_name="train_content")
    dev_df = pd.read_excel(data_path, header=0, sheet_name="dev_content")
    for txt,label in zip(train_df['Column6'].tolist(),train_df['Column7'].tolist()):
        res_list.append(dict(text=txt, label=label))
    for txt,label in zip(dev_df['Column6'].tolist(),dev_df['Column7'].tolist()):
        dedv_list.append(dict(text=txt, label=label))
    return res_list,dedv_list


#
# train_ds, dev_ds, test_ds = SelfDefinedDataset.get_datasets([trainlst, devlst, testlst])
if __name__ == "__main__":
    trainlst = SelfDefinedDataset('data/train.xlsx', "train_content")
    print(trainlst.data)
    # train_ds, dev_ds = SelfDefinedDataset.get_datasets([trainlst, devlst])