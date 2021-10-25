# -*- coding: UTF-8 -*-


import json
import csv
import re


# 规范化context字段，去除其中的换行符
def context_modifier(context):
    index = context.find('\n')
    while index != -1:
        context = context.replace(context[index], '')
        index = context.find('\n')


# 提取json文件中的label和text标签，生成新的csv
def csv_writer():
    file = open('./csv/exhibits.csv', 'w', encoding='utf-8')
    paras = json.load(open('./json/Training_Vegetable.json', 'r', encoding='utf-8'))
    para_num = len(paras)
    for i in range(para_num):
        context = paras[i]['paragraphs'][0]['context']
        context_modifier(context)
        label = paras[i]['title']
        file.write('"' + label + '"')
        file.write(',')
        file.write('"' + context + '"')
        file.write('\n')
    file.close()


# 读取已标注的QA，填入json文件中的空缺并生成新的json文件
def json_handler():
    # 读取csv，存入列表
    data = csv.reader(open('./csv/QA.csv', 'r', encoding='utf-8'))
    qas = []
    for line in data:
        qas.append(line)
    qas.pop(0)

    #
    paras = json.load(open('./json/Training_Vegetable.json', 'r', encoding='utf-8'))
    para_num = len(paras)
    for i in range(para_num):
        question_cnt = 0
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
    outcome = open('./json/outcome.json', 'w', encoding='utf-8')
    json.dump(paras, outcome, ensure_ascii=False, indent=3)


if __name__ == '__main__':
    json_handler()
