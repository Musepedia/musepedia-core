# -*- coding: UTF-8 -*-

import json
import csv
import re


# 提供CSV文件的路径，每一行作为一个元素，存入列表并返回
# CSV文件存入./csv/目录下
# 默认encoding为GBK，可手动设置为UTF-8
def csv_reader(path, encoding='gbk'):
    data = csv.reader(open(path, 'r', encoding=encoding))
    list_csv = []
    for line in data:
        list_csv.append(line)
    list_csv.pop(0)
    return list_csv


# 处理json文件，写入新的json文件中
# 待处理的json文件与输出的json文件均在./json/目录下
# 输入两个文件的路径及其对应编码，默认为UTF-8
# 无返回值
def json_handler(file1, file2, encoding1='utf-8', encoding2='utf-8'):
    json1 = json.load(open(file1, 'r', encoding=encoding1))
    json2 = open(file2, 'w', encoding=encoding2)

    paras = json1
    para_num = len(paras)
    for i in range(para_num):
        question_cnt = 0
        for qa in qas:
            qid = qa[0]
            if int(qid) == i:
                id = "TRAIN_"+qid+"_QUERY_"+str(question_cnt)
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


if __name__=='__main__':

    qas = csv_reader('./csv/QA.csv')

    # 待填入qas数据的json
    filepath1 = './json/Botany.json'
    # 运行产生的json
    filepath2 = './json/Outcome.json'
    json_handler(filepath1, filepath2)

