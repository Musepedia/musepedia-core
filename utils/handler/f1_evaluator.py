# -*- coding: UTF-8 -*-


import os
import sys
import jieba
import utils.handler.JsonHandler as JsonHandler

# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import src.qa.core.QuestionAnswering as QA

def get_input():
    qas = JsonHandler.qa_list()
    texts = JsonHandler.get_texts()
    truth_list = []
    answer_list = []
    for qa in qas:
        qid = qa[0]
        text = texts[int(qid)]
        question = qa[1]
        answer_list.append(QA.get_answer(question, text))
        truth_list.append(qa[2])
    return answer_list, truth_list


def stop_word(word_list):
    stop_list = ["的", "地", "得"]

    for word in word_list:
        if word in stop_list:
            word_list.pop(word_list.index(word))


def evaluator():
    # Plist = []
    # Rlist = []
    # F1list = []
    answers, truths = get_input()
    file = open('./csv/evaluation.csv', 'w', encoding='utf-8')
    file.write('text_id,answer,truth,f1_score\n')
    for i in range(len(answers)):
        file.write('"'+ str(i) + '"' + ',')
        answer = answers[i]
        file.write('"' + answer + '"' + ',')
        truth = truths[i]
        file.write('"' + truth + '"' + ',')
        truth_word_list = jieba.lcut(truth, cut_all=False)
        answer_word_list = jieba.lcut(answer, cut_all=False)
        stop_word(truth_word_list)
        stop_word(answer_word_list)

        truth_len = len(truth_word_list)
        answer_len = len(answer_word_list)

        word_cnt = set(truth_word_list) & set(answer_word_list)

        TP = len(word_cnt)
        if not answer_len:
            precision = 0
        else:
            precision = TP / answer_len
        # Plist.append(precision)
        if not truth_len:
            recall = 0
        else:
            recall = TP / truth_len
        # Rlist.append(recall)
        if precision == 0 and recall == 0:
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        file.write(str(F1) + '\n')
        # F1list.append(F1)


    # return Plist, Rlist, F1list


if __name__ == '__main__':
    evaluator()
    # P, R, F1 = evaluator()
    # print(F1)