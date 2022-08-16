# Author:xyy
# CreatTime:2022-07-10
# FileName:convert_format
# Description: 修改比赛给定测试文件格式为代码所需格式

def process_test_data(file_name):
    with open('./data/raw_data/' + file_name, encoding='utf-8') as file:
        file_content = eval(file.read().strip())

    with open('./data/processed_data/' + file_name, 'a', encoding='utf-8') as file:
        content_text = {}
        for item in file_content:
            data_text = item['my_text']
            content_text["id"] = item["id"]
            content_text["text"] = data_text.strip()
            file.write(f"{content_text}\n")


process_test_data('test.json')
