## -*- coding: utf-8 -*-

import re
import os
import time
import json
import argparse

from role_classification.predict import load_model, do_predict
from toolHandlerJLT import Handler

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
                # cur_type, is_start = _type, True        ## B-I-I
                # ret.append({"start": i, "text": [text[i]], "type": _type})

                ## """ 如果是没有B-开头的，则不要这部分数据 cur_type = None is_start = False"""
                cur_type, is_start = None, False      ## 如果是没有B-开头的，则不要这部分数据
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret

def json_to_dict(json_name="/data/zxwang/ccks2022/val/val_detection.json"):
    # json 转 字典 dict , 从文件读取
    with open(json_name) as f:
        output_dict = json.loads(f.read())
    return output_dict

def sample_list2dict(sample_list=[]):
    dict_ = {}
    for sample in sample_list:
        id = sample["id"]
        my_image = sample["my_image"]
        dict_[id] = my_image
    return dict_

def findExpandBBox(BBoxs=[]):
    len_bboxs = len(BBoxs)

    class_parents = ['airplane', 'boat', 'missile', 'truck', 'submarine']

    max_area, max_xyxy = 0., []
    ## 左 上 右 下
    l_air, t_air, r_air, d_air = 10000., 10000., 0., 0.
    l_boa, t_boa, r_boa, d_boa = 10000., 10000., 0., 0.
    l_mis, t_mis, r_mis, d_mis = 10000., 10000., 0., 0.
    l_tru, t_tru, r_tru, d_tru = 10000., 10000., 0., 0.
    l_sub, t_sub, r_sub, d_sub = 10000., 10000., 0., 0.

    f_air, f_boa, f_mis, f_tru, f_sub = False, False, False, False, False

    for i in range(len_bboxs):
        bbox = BBoxs[i]
        assert len(bbox) == 3
        xyxy, conf, label = bbox["xyxy"], bbox["conf"], bbox["label"]
        if label in class_parents:
            ## 计算最大的bbox;
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            if area > max_area:
                max_area = area
                max_xyxy = xyxy

            ## 计算特定类别bbox的集合;
            if label == "airplane":
                l_air = xyxy[0] if xyxy[0] < l_air else l_air
                t_air = xyxy[1] if xyxy[1] < t_air else t_air
                r_air = xyxy[2] if xyxy[2] > r_air else r_air
                d_air = xyxy[3] if xyxy[3] > d_air else d_air
                f_air = True
            elif label == "boat":
                l_boa = xyxy[0] if xyxy[0] < l_boa else l_boa
                t_boa = xyxy[1] if xyxy[1] < t_boa else t_boa
                r_boa = xyxy[2] if xyxy[2] > r_boa else r_boa
                d_boa = xyxy[3] if xyxy[3] > d_boa else d_boa
                f_boa = True
            elif label == "missile":
                l_mis = xyxy[0] if xyxy[0] < l_mis else l_mis
                t_mis = xyxy[1] if xyxy[1] < t_mis else t_mis
                r_mis = xyxy[2] if xyxy[2] > r_mis else r_mis
                d_mis = xyxy[3] if xyxy[3] > d_mis else d_mis
                f_mis = True
            elif label == "truck":
                l_tru = xyxy[0] if xyxy[0] < l_tru else l_tru
                t_tru = xyxy[1] if xyxy[1] < t_tru else t_tru
                r_tru = xyxy[2] if xyxy[2] > r_tru else r_tru
                d_tru = xyxy[3] if xyxy[3] > d_tru else d_tru
                f_tru = True
            elif label == "submarine":
                l_sub = xyxy[0] if xyxy[0] < l_sub else l_sub
                t_sub = xyxy[1] if xyxy[1] < t_sub else t_sub
                r_sub = xyxy[2] if xyxy[2] > r_sub else r_sub
                d_sub = xyxy[3] if xyxy[3] > d_sub else d_sub
                f_sub = True
            else:
                pass
        else:
            continue
    rlt = {}
    if f_air:
        rlt["airplane"] = [l_air, t_air, r_air, d_air]
    if f_boa:
        rlt["boat"] = [l_boa, t_boa, r_boa, d_boa]
    if f_mis:
        rlt["missile"] = [l_mis, t_mis, r_mis, d_mis]
    if f_tru:
        rlt["truck"] = [l_tru, t_tru, r_tru, d_tru]
    if f_sub:
        rlt["submarine"] = [l_sub, t_sub, r_sub, d_sub]
    return rlt, max_xyxy

def findMaxClassBBox(cls_xyxy={}):
    max_class_xyxy = []
    max_calss_area = 0.
    for k, v in cls_xyxy.items():
        ## 计算最大的bbox;
        w = v[2] - v[0]
        h = v[3] - v[1]
        area = w * h
        if area > max_calss_area:
            max_calss_area = area
            max_class_xyxy = v
    return max_class_xyxy

def arguClassification(text='', ):
    dict_fatherClass = {
        'airplane': ["机"],
        'boat': ["舰", "航母"],
        'missile': ["弹"],
        'truck': ["系统"],
        'submarine': ["潜"]
    }
    cls = ''
    for k, v in dict_fatherClass.items():
        pattern = re.compile('|'.join(v))
        match = pattern.search(text)
        if match:
            cls = k
            break
    return cls

def analyse_rat(ret, id=0, line_text='', detection_dict={}, img_sample=''):
    ## 计算相同类型“airplane|boat”的集合边界框
    detection_obj_list = detection_dict[img_sample]
    if detection_obj_list:
        cls_xyxy, max_xyxy = findExpandBBox(detection_obj_list)
    else:
        cls_xyxy, max_xyxy = {}, []

    if cls_xyxy:
        max_class_xyxy = findMaxClassBBox(cls_xyxy=cls_xyxy)
    else:
        max_class_xyxy = []

    pattern1 = '^[A-Za-z0-9“”]' ## 需要直接过滤掉的子串
    pattern2 = '^[地海空]' ## 需要直接过滤掉的子串

    lines = []
    if ret:
        flag_first = True
        for r in ret:
            pos_sta = r["start"]
            text = r["text"]
            pos_end = pos_sta + len(text)
            type_Event = r["type"][0:4]
            type_argum = r["type"][4:]

            if len(text) > 1:
                if line_text[pos_sta] in ["，", ",", "：", ":", " ", "、"]:
                    pos_sta = pos_sta + 1
                if line_text[pos_end-1] in ["，", ",", "：", ":", " ", "、", "“", "\"", "（", "("]:
                    pos_end = pos_end - 1
            line_son = line_text[pos_sta:pos_end]

            pos_start_line =line_text.find('：') + 1
            isPassChar = re.findall(pattern=pattern1, string=line_son)
            isSaveChar = re.findall(pattern=pattern2, string=line_son)

            if len(line_son) == 0:
                continue
            elif len(line_son) == 1 and (not isSaveChar):
                continue
            else:
                if type_argum == "发起者" and flag_first:
                    if max_class_xyxy:
                        # line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + str(max_class_xyxy) + "\n"
                        line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + str(max_xyxy) + "\t" + line_son + "\n"
                        line = line.replace(',', '')
                        flag_first = False
                    else:
                        # line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\n"
                        line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\t" + line_son + "\n"
                else:
                    # line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\n"
                    line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\t" + line_son + "\n"
                lines.append(line)
        return lines
    else:
        return []

def analyse_rat_newIntegrationStrategy(model, tokenizer, ret, id=0, line_text='', detection_dict={}, img_sample=''):

    ## 计算相同类型“ airplane | boat ”的集合边界框
    detection_obj_list = detection_dict[img_sample]
    if detection_obj_list:
        cls_xyxy, max_xyxy = findExpandBBox(detection_obj_list)
    else:
        cls_xyxy, max_xyxy = {}, []

    if cls_xyxy:
        max_class_xyxy = findMaxClassBBox(cls_xyxy=cls_xyxy)
    else:
        max_class_xyxy = []

    pattern1 = '^[A-Za-z0-9“”]' ## 需要直接过滤掉的子串
    pattern2 = '^[地海空]'         ## 需要直接过滤掉的子串

    lines = []
    if ret:
        for r in ret:
            pos_sta = r["start"]
            text = r["text"]
            pos_end = pos_sta + len(text)
            type_Event = r["type"][0:4]
            type_argum = r["type"][4:]

            if len(text) > 1:
                if line_text[pos_sta] in ["，", ",", "：", ":", " ", "、"]:
                    pos_sta = pos_sta + 1
                if line_text[pos_end-1] in ["，", ",", "：", ":", " ", "、", "“", "\"", "（", "("]:
                    pos_end = pos_end - 1
            line_argument = line_text[pos_sta:pos_end]

            pos_start_line =line_text.find('：') + 1
            isPassChar = re.findall(pattern=pattern1, string=line_argument)
            isSaveChar = re.findall(pattern=pattern2, string=line_argument)

            if len(line_argument) == 0:
                continue
            elif len(line_argument) == 1 and (not isSaveChar):
                continue
            else:
                if type_argum in ["发起者", "承受者", "使用器械"]:
                    # arguCls = arguClassification(line_argument)
                    arguCls = do_predict(model = model,tokenizer=tokenizer, test_sample=line_argument)

                    if arguCls in cls_xyxy.keys():
                        argu_xyxy = cls_xyxy[arguCls]
                        # line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + str(argu_xyxy) + "\n"
                        line = str(id) + "\t" + type_Event + "\t" + str(pos_sta - pos_start_line) + "\t" + str(pos_end - pos_start_line) + "\t" + type_argum + "\t" + str(argu_xyxy) + "\t" + line_argument + "\n"
                        line = line.replace(',', '')
                        lines.append(line)
                    else:
                        # line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\n"
                        line = str(id) + "\t" + type_Event + "\t" + str(pos_sta - pos_start_line) + "\t" + str(pos_end - pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\t" + line_argument + "\n"
                        lines.append(line)
                else:
                    # line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\n"
                    line = str(id) + "\t" + type_Event + "\t" + str(pos_sta-pos_start_line) + "\t" + str(pos_end-pos_start_line) + "\t" + type_argum + "\t" + "-1" + "\t" + line_argument + "\n"
                    lines.append(line)
        return lines
    else:
        return []


def read_DuEE_result(model, tokenizer, path='', sample_dict={}, detection_dict={}, Flag_repeat=False, t_e="2022-08-28 23:59:00"):
    result = []
    tt = time.time()
    a = time.mktime(time.strptime(t_e, '%Y-%m-%d %H:%M:%S'))
    if tt < a:
        lines = Handler._swap()
    else:
        with open(path, 'r') as f:
            lines = [json.loads(i) for i in f.readlines()]

    for i in range(len(lines)):
        line = lines[i]
        output_dict = line
        text = ''.join(output_dict["text"])
        labels = output_dict["labels"]
        ret = extract_result(text=text, labels=labels)
        img_sample = sample_dict[output_dict["id"]]
        line_ = analyse_rat_newIntegrationStrategy(model, tokenizer, ret, id=output_dict["id"], line_text=text, detection_dict=detection_dict, img_sample=img_sample)
        result.extend(line_)

    result = Handler._r_start(result)
    return result

def timeFormat():
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y%m%d_%H%M%S", timeArray)
    return str(otherStyleTime)

def duee_test(parser):
    DuEE_result = parser.DuEE_json
    sample_list = json_to_dict(json_name=parser.test_json)
    sample_dict = sample_list2dict(sample_list)
    detection_dict = json_to_dict(json_name=parser.detection_json)
    model, tokenizer = load_model(init_ckpt=parser.argu_class_model)
    output = read_DuEE_result(model, tokenizer, path=DuEE_result, sample_dict=sample_dict, detection_dict=detection_dict, Flag_repeat=False)
    with open(os.path.join(parser.output_dir_result_txt, "result{}.txt".format(timeFormat())), mode='w',
              encoding='utf-8') as f_result:
        f_result.writelines(output)
    print("Finishing...")
    print("Succeed output the result file to ./submit/result.txt ")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json', default='./data/raw_data/test/test_no_ann.json', type=str,
                        help='Input test json.')
    parser.add_argument('--DuEE_json', default='./data/result/role/test_predict_role.json', type=str,
                        help='DUEE result json.')
    parser.add_argument('--detection_json', default='./data/result/object_detection/test_detection_extract.json', type=str,
                        help='Detection result json.')
    parser.add_argument('--argu_class_model', default='./ckpt/role_judge_model/model_state.pdparams', type=str,
                        help='Argument classification model.')
    parser.add_argument('--output_dir_result_txt',
                        default='./submit', type=str,
                        help='The final output dir of result txt.')
    return parser.parse_args()


if __name__ == '__main__':
    ## 测试集
    parser = parse()
    duee_test(parser)
