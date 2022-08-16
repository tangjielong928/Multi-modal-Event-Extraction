import os
import sys
import json
from utils import read_by_lines, write_by_lines,text_to_sents,cal_md5

enum_role = '填充'

def data_process(path, model="trigger", is_predict=False):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = "B-" if i == start else "I-"
            data[i] = "{}{}".format(suffix, _type)
        return data

    sentences = []
    output = ["text_a"] if is_predict else ["text_a\tlabel"]
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip())
            _id = d_json["id"]
            text_a = [
                "，" if t == " " or t == "\n" or t == "\r" else t
                for t in list(d_json["text"].lower())
            ]
            if is_predict:
                sentences.append({"text": d_json["text"], "id": _id})
                output.append('\002'.join(text_a))
            else:
                if model == "trigger":
                    #过滤空事件
                    if len(d_json.get("event_list", [])) > 0:
                        labels = ["O"] * len(text_a)
                        for event in d_json.get("event_list", []):

                            event_type = event["event_type"]
                            trigger = event["trigger"]
                            try:
                                start = d_json["text"].find(trigger)
                                labels = label_data(labels, start, len(trigger),
                                                    event_type)
                            except:
                                pass
                        output.append("{}\t{}".format('\002'.join(text_a),
                                                          '\002'.join(labels)))
                elif model == "role":
                    if len(d_json.get("event_list", []))>0:
                        for event in d_json.get("event_list", []):
                            labels = ["O"] * len(text_a)
                            event_type = event["event_type"]
                            trigger = event["trigger"]
                            #将事件类型+trigger拼接到text开头构成先验信息
                            txt_event = event_type+trigger+'：'
                            for arg in event["arguments"]:
                                role_type = arg["role"]
                                argument = arg["argument"]
                                try:
                                    start = arg["argument_start_index"]
                                    labels = label_data(labels, start, len(argument),
                                                        role_type)
                                except:
                                    pass
                            txt = [
                                "，" if t == " " or t == "\n" or t == "\r" else t
                                for t in list(txt_event.lower())
                            ]
                            txt_event_label = ["O"] * len(txt_event)+labels

                            output.append("{}\t{}".format('\002'.join(txt + text_a),
                                                          '\002'.join(txt_event_label)))
                    # else:
                    #     labels = ["O"] * len(text_a)
                    #     output.append("{}\t{}".format('\002'.join(text_a),
                    #                                       '\002'.join(labels)))

    return output


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
    tags = []
    for index, label in enumerate(labels):
        tags.append("{}\t{}".format(index, label))
    return tags

# if __name__ == "__main__":
#     train_sent = docs_data_process(
#         "./data/EE1.0/duee_train.json")
#     print(train_sent)


if __name__ == "__main__":
    print("\n=================CCKS 1.0 DATASET==============")
    conf_dir = "../data/EE1.0"
    # conf_dir = "./conf/Event_extraction-fin-1"
    schema_path = "{}/event_schema.json".format(conf_dir)
    tags_trigger_path = "{}/trigger_tag.dict".format(conf_dir)
    tags_role_path = "{}/role_tag.dict".format(conf_dir)
    print("\n=================start schema process==============")
    print('input path {}'.format(schema_path))
    tags_trigger = schema_process(schema_path, "trigger")
    write_by_lines(tags_trigger_path, tags_trigger)
    print("save trigger tag {} at {}".format(len(tags_trigger),
                                             tags_trigger_path))
    tags_role = schema_process(schema_path, "role")
    write_by_lines(tags_role_path, tags_role)
    print("save role tag {} at {}".format(len(tags_role), tags_role_path))
    print("=================end schema process===============")

    # data process
    data_dir = "../data/EE1.0"
    # data_dir = "./data/Event_extraction-fin-1"
    trigger_save_dir = "{}/trigger".format(data_dir)
    role_save_dir = "{}/role".format(data_dir)
    print("\n=================start schema process==============")
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)
    if not os.path.exists(role_save_dir):
        os.makedirs(role_save_dir)
    print("\n----trigger------for dir {} to {}".format(data_dir,
                                                       trigger_save_dir))
    train_tri = data_process("{}/train.json".format(data_dir), "trigger")
    write_by_lines("{}/train.tsv".format(trigger_save_dir), train_tri)
    dev_tri = data_process("{}/dev.json".format(data_dir), "trigger")
    write_by_lines("{}/dev.tsv".format(trigger_save_dir), dev_tri)
    test_tri = data_process("{}/test.json".format(data_dir), "trigger")
    write_by_lines("{}/test.tsv".format(trigger_save_dir), test_tri)
    print("train {} dev {} test {}".format(len(train_tri), len(dev_tri),
                                           len(test_tri)))
    print("\n----role------for dir {} to {}".format(data_dir, role_save_dir))
    train_role = data_process("{}/train.json".format(data_dir), "role")
    write_by_lines("{}/train.tsv".format(role_save_dir), train_role)
    dev_role = data_process("{}/dev.json".format(data_dir), "role")
    write_by_lines("{}/dev.tsv".format(role_save_dir), dev_role)
    test_role = data_process("{}/test.json".format(data_dir), "role")
    write_by_lines("{}/test.tsv".format(role_save_dir), test_role)
    print("train {} dev {} test {}".format(len(train_role), len(dev_role),
                                           len(test_role)))
    print("=================end schema process==============")