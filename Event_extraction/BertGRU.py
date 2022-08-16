# -*- coding:utf-8 -*-
# @Time : 2022/7/18 15:54
# @Author: Jielong Tang
# @File : BertGRU.py
'''
在预训练模型基础上增加两层双向GRU+CRF用于序列标注
'''

from paddlehub.common.utils import version_compare
import paddle
import os
import json
import numpy as np
from utils import read_by_lines
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if "B-{}".format(_type) not in labels:
            labels.extend(["B-{}".format(_type), "I-{}".format(_type)])
        return labels

    labels = []
    for line in read_by_lines(path):
        d_json = json.loads(line.strip())
        if model == "trigger":
            labels = label_add(labels, d_json["event_type"])
        elif model == "role":
            for role in d_json["role_list"]:
                labels = label_add(labels, role["role"])
    labels.append("O")
    return labels

class SequenceLabelTaskSP(hub.SequenceLabelTask):
    '''
    扩展序列标注任务
    增加从非best_model目录加载模型的功能
    添加gru层
    '''

    def __init__(self,
                 feature,
                 max_seq_len,
                 num_classes,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 metrics_choices="default",
                 add_crf=False):

        print("SequenceLabelTaskSP")

        super(SequenceLabelTaskSP, self).__init__(
            feature=feature,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
            feed_list=feed_list,
            data_reader=data_reader,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices,
            add_crf=add_crf)

    # def init_if_load_best_model(self):
    #     '''
    #     支持从自定义的目录加载bestmodel
    #     '''
    #     if not self.is_best_model_loaded:
    #         best_model_path = os.path.join(self.config.checkpoint_dir,
    #                                        args.predictmodel)
    #         logger.info("Load the best model from %s" % best_model_path)
    #         if os.path.exists(best_model_path):
    #             self.load_parameters(best_model_path)
    #             self.is_checkpoint_loaded = False
    #             self.is_best_model_loaded = True
    #         else:
    #             self.init_if_necessary()
    #     else:
    #         logger.info("The best model has been loaded")

    def _build_net(self):
        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64', lod_level=0)

        if version_compare(paddle.__version__, "1.6"):
            self.seq_len_used = fluid.layers.squeeze(self.seq_len, axes=[1])
        else:
            self.seq_len_used = self.seq_len

        # 增加gru层相关的代码
        grnn_hidden_dim = 256  # 768
        crf_lr = 0.2
        bigru_num = 2
        init_bound = 0.1

        def _bigru_layer(input_feature):
            """define the bidirectional gru layer
            """
            pre_gru = fluid.layers.fc(
                input=input_feature,
                size=grnn_hidden_dim * 3,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            gru = fluid.layers.dynamic_gru(
                input=pre_gru,
                size=grnn_hidden_dim,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            pre_gru_r = fluid.layers.fc(
                input=input_feature,
                size=grnn_hidden_dim * 3,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            gru_r = fluid.layers.dynamic_gru(
                input=pre_gru_r,
                size=grnn_hidden_dim,
                is_reverse=True,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
            return bi_merge

        if self.add_crf:
            unpad_feature = fluid.layers.sequence_unpad(
                self.feature, length=self.seq_len_used)

            # 增加gru层相关的代码
            input_feature = unpad_feature
            for i in range(bigru_num):
                bigru_output = _bigru_layer(input_feature)
                input_feature = bigru_output

            unpad_feature = input_feature
            self.emission = fluid.layers.fc(
                size=self.num_classes,
                input=unpad_feature,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(low=-0.1, high=0.1),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            size = self.emission.shape[1]
            fluid.layers.create_parameter(
                shape=[size + 2, size], dtype=self.emission.dtype, name='crfw')
            self.ret_infers = fluid.layers.crf_decoding(
                input=self.emission, param_attr=fluid.ParamAttr(name='crfw'))
            ret_infers = fluid.layers.assign(self.ret_infers)
            return [ret_infers]
        else:
            self.logits = fluid.layers.fc(
                input=self.feature,
                size=self.num_classes,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(
                    name="cls_seq_label_out_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name="cls_seq_label_out_b",
                    initializer=fluid.initializer.Constant(0.)))

            self.ret_infers = fluid.layers.reshape(
                x=fluid.layers.argmax(self.logits, axis=2), shape=[-1, 1])

            logits = self.logits
            logits = fluid.layers.flatten(logits, axis=2)
            logits = fluid.layers.softmax(logits)
            self.num_labels = logits.shape[1]
            return [logits]


class EEDataset(BaseNLPDataset):
    """EEDataset"""

    def __init__(self, data_dir, labels, model="trigger"):
        pdf = "{}/test.tsv".format(model)
        # if not args.predict_data:
        #     pdf = "{}_test.tsv".format(model)
        print("labels:", labels)
        # 数据集存放位置
        super(EEDataset, self).__init__(
            base_path=data_dir,
            train_file="{}/train.tsv".format(model),
            dev_file="{}/dev.tsv".format(model),
            test_file="{}/test.tsv".format(model),
            # 如果还有预测数据（不需要文本类别label），可以放在predict.tsv
            predict_file=pdf,
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            predict_file_with_header=True,
            # 数据集类别集合
            label_list=labels)

def do_train(max_seq_len,data_dir,schema_labels,do_model,weight_decay,learning_rate,use_gpu,eval_step,model_save_step,num_epoch,batch_size,
             checkpoint_dir,add_gru = True, add_crf=True, _do_train=True, _do_eval=True):
    model_name = "bert-base-uncased"
    # model_name = "chinese-roberta-wwm-ext-large"
    module = hub.Module(name=model_name)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=max_seq_len)

    # Download dataset and use SequenceLabelReader to read dataset
    dataset = EEDataset(data_dir, schema_labels, model=do_model)
    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name
    ]

    # Select a finetune strategy
    #默认值warmup_proportion = 0.1
    warmup_proportion = 0.1
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=warmup_proportion,
        weight_decay=weight_decay,
        learning_rate=learning_rate)

    print("use_cuda:", use_gpu)

    # Setup runing config for PaddleHub Finetune API
    use_data_parallel = True
    config = hub.RunConfig(
        eval_interval=eval_step,
        save_ckpt_interval=model_save_step,
        use_data_parallel=use_data_parallel,
        use_cuda=use_gpu,
        num_epoch=num_epoch,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        strategy=strategy)

    # Define a sequence labeling finetune task by PaddleHub's API
    # If add crf, the network use crf as decoder

    print("add_gru", add_gru)
    print("add_crf", add_crf)

    if add_gru:

        seq_label_task = SequenceLabelTaskSP(
            data_reader=reader,
            feature=sequence_output,
            feed_list=feed_list,
            max_seq_len=max_seq_len,
            num_classes=dataset.num_labels,
            config=config,
            add_crf=add_crf)
    else:
        seq_label_task = hub.SequenceLabelTask(
            data_reader=reader,
            feature=sequence_output,
            feed_list=feed_list,
            max_seq_len=max_seq_len,
            num_classes=dataset.num_labels,
            config=config,
            add_crf=add_crf)

    # 创建 LogWriter 对象
    # log_writer = MyLog(mode="role2")
    # seq_label_task._tb_writer = log_writer

    # Finetune and evaluate model by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    if _do_train:
        print("start finetune and eval process")
        seq_label_task.finetune_and_eval()
        seq_label_task.best_score = -999

    if _do_eval:
        print("start eval process")
        seq_label_task.eval()


if __name__ == "__main__":
    #训练role序列标注
    schema_labels = schema_process(path='../data/EE1.0/duee_fin_event_schema.json', model='role')
    do_train(max_seq_len = 300, data_dir ='../data/EE1.0/role', schema_labels= schema_labels, do_model ='role', weight_decay = 0.01, learning_rate = 5e-5, use_gpu = True,
             eval_step = 50,
             model_save_step = 200,
             num_epoch = 50,
             batch_size = 16,
             checkpoint_dir ='../ckpt/EE1.0/role-ENRINE+GRU/',
             add_gru = True, add_crf=True, _do_train=True, _do_eval=True)
    # print(schema_process(path='./data/EE1.0/event_schema.json',model='role'))