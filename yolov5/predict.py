# -*- coding: utf-8 -*-

import json
import argparse
import detect as detect
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

def dict_to_json(input_dict, json_name="/data/zxwang/ccks2022/val/val_detection.json"):
    # 字典 dict 转 json, 写入文件
    with open(json_name, "w") as f:
        f.write(json.dumps(input_dict, indent=4))

def predict_detection(parser):
    detect_api = detect.DetectAPI_Img(
        weights=parser.best_model,
        project='runs/detect',
        name='myexp',
        exist_ok=True,
        line_thickness=2,
        save_txt=False,
        save_bbox_p_label=True,
        nosave=True,
        )
    result = detect_api.run_img(imgSource=parser.input_img_source)
    dict_to_json(result, json_name=parser.output_detection_json)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_source', default='../data/raw_data/test/test_images/', type=str,
                        help='The dir of input image.')
    parser.add_argument('--best_model', default='../ckpt/yolov5/best.pt', type=str,
                        help='The best model from fine-tuning yolov5l on ccks2022 img dataset.')
    parser.add_argument('--output_detection_json', default="../data/result/object_detection/test_detection_extract1.json", type=str,
                        help='the output json of object detection.')
    return parser.parse_args()

if __name__ == '__main__':
    parser = parse()
    predict_detection(parser)


