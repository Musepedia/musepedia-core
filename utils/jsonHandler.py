# -*- coding: UTF-8 -*-

import json
import csv
import re


def main(path_csv1, path_csv2, path_json1, path_json2, csv_encoding='gbk', json_encoding1='utf-8', json_encoding2='utf-8'):
    # 读取csv，存入列表
    data = csv.reader(open(path_csv1, 'r', encoding=csv_encoding))
    qas = []
    for line in data:
        qas.append(line)
    qas.pop(0)
    
    # 处理json
    json1 = json.load(open(path_json1, 'r', encoding=json_encoding1))
    json2 = open(path_json2, 'w', encoding=json_encoding2)

    paras = json1
    para_num = len(paras)
    # print(paras)

    text_list = []
    label_list = []
    for i in range(para_num):
        question_cnt = 0
        # 规范化换行符
        text = paras[i]['paragraphs'][0]['context']
        index = text.find('\n')
        while index != -1:
            text = text.replace(text[index], '')
            index = text.find('\n')
        text_list.append(text)
        label = paras[i]['title']
        label_list.append(label)
        for qa in qas:
            qid = qa[0]
            if int(qid) == i:
                id = "TRAIN_" + qid + "_QUERY_" + str(question_cnt)
                question = qa[1]
                answer = qa[2]
                # print(answer)
                line = paras[i]['paragraphs'][0]['context']
                # print(line)
                answer_start = re.search(answer, line).start()
                ans_dict = {'text': answer, 'answer_start': answer_start}
                ret_dict = {'question': question, 'id': id, 'answers': [ans_dict]}
                question_cnt += 1
                paras[i]['paragraphs'][0]['qas'].append(ret_dict)
    json.dump(json1, json2, ensure_ascii=False, indent=3)
    # 将label和text写入新csv
    csv_2 = open(path_csv2, 'w', encoding='utf-8')
    csv_2.write('label,text\n')
    for i in range(len(label_list)):
        csv_2.write(label_list[i])
        csv_2.write(',')
        csv_2.write(text_list[i])
        csv_2.write('\n')


if __name__ == '__main__':
    csvpath1 = './QA.csv'
    csvpath2 = './exhibits.csv'
    filepath1 = './Botany.json'
    filepath2 = './Outcome.json'
    main(csvpath1, csvpath2, filepath1, filepath2)




