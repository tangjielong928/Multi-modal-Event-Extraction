# -*- coding:utf-8 -*-
# @Time : 2022/7/20 17:07
# @Author: Jielong Tang
# @File : BERT-gru-crf.py
import paddle
import paddle.nn as nn
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder
from paddlenlp.transformers import ErnieModel,BertModel

class FGM():
    """针对embedding层梯度上升干扰的对抗训练方法,Fast Gradient Method（FGM）"""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:  # 检验参数是否可训练及范围
                # print('begin')
                self.backup[name] = param.numpy()  # 备份原有参数值
                grad_tensor = paddle.to_tensor(param.grad)  # param.grad是个numpy对象
                norm = paddle.norm(grad_tensor)  # norm化
                if norm != 0:
                    r_at = epsilon * grad_tensor / norm
                    param.add(r_at)  # 在原有embed值上添加向上梯度干扰

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])  # 将原有embed参数还原
        self.backup = {}

class ptm_GRUCRF(nn.Layer):
    def __init__(self,pretrain_model,num_class,gru_hidden_size=300,
                crf_lr=100):
        super().__init__()
        self.num_classes = num_class
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.gru = nn.GRU(self.bert.config["hidden_size"],
                          gru_hidden_size,
                          num_layers = 2,
                          direction='bidirect')
        self.fc = nn.Linear(gru_hidden_size*2,num_class)
        self.crf = LinearChainCrf(self.num_classes)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self,input_ids,token_type_ids,lengths=None,labels=None):
        encoder_output,_ = self.bert(input_ids, token_type_ids = token_type_ids,output_hidden_states = False)
        # all_hidden_states = paddle.stack(encoder_output,axis=)
        # concatenate_pooling = paddle.concat(
        #     (encoder_output[-1], encoder_output[-2], encoder_output[-3], encoder_output[-4]), -1
        # )
        # print(concatenate_pooling)
        # concatenate_pooling = concatenate_pooling[:, 0]
        gru_output, _ = self.gru(encoder_output)
        emission = self.fc(gru_output)
        if labels is not None:
            loss = self.crf_loss(emission, lengths, labels)
            return loss
        else:
            _,prediction = self.viterbi_decoder(emission, lengths)
            return prediction

if __name__ =='__main__':
    print('01')