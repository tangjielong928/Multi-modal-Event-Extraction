# Author:xyy
# CreatTime:
# FileName:convert_format
# Description: 修改比赛给定训练\验证文件格式为代码所需格式

def process_data(file_name):
    with open('./data/raw_data/' + file_name, encoding='utf-8') as file:
        file_content = eval(file.read().strip())

    with open('./data/processed_data/' + file_name, 'a', encoding='utf-8') as file:
        content_text = {}
        for item in file_content:
            data_text = item['my_text']
            content_text['text'] = data_text.strip()
            content_text['event_list'] = []
            data_event = item['events']
            if len(data_event) > 0:
                for i in data_event:
                    content_text['event_list'].append({'event_type': i[1]})
                if len(content_text['event_list']) >= 0:
                    file.write(f"{content_text}\n")
            elif len(data_event) == 0:
                content_text['event_list'].append({'event_type': '无事件'})
                file.write(f"{content_text}\n")


process_data('train.json')
# process_data('dev.json')
