# -*- coding: UTF-8 -*-
import torch
import numpy as np


from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from common.exception.ExceptionHandler import catch, check_length
from common.exception.TextLengthError import TextLengthError
from common.log.QALogging import qa_logging
from src.qa.core.Answer import Answer

MODEL_PATH = 'src/qa/models/roberta-base-chinese-extractive-qa'


def preload():
    """
    根据模型存储地址，加载tokenizer和模型
    :return: tokenizer与模型
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

    return tokenizer, model


def get_pos_with_logit(start_logits, end_logits):
    """
    计算答案的起始位置和终止位置，以及相应的logits
    :param start_logits: 起始logits
    :param end_logits: 终止logits
    :return: 2个tuple (position, logits)分别代表答案起始位置和终止位置
    """

    startPosWithLogits = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    endPosWithLogits = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

    return startPosWithLogits[0], endPosWithLogits[0]


@catch(TextLengthError, MemoryError, AttributeError)
@check_length(512)
@qa_logging(question_index=2, text_index=3)
def get_possible_answer(tokenizer, model, question, text):
    """
    处理1个问题对应1篇文本，并得到答案
    :param tokenizer: 用于将字符与token映射
    :param model: 模型（现在是Roberta）
    :param question: 问题
    :param text: 待抽取的文本
    :return: 问题对应的答案，如果没有答案，那么返回[CLS]
    """

    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
    outputs = model(**inputs)

    answerStartLogits = outputs.start_logits[0]
    answerEndLogits = outputs.end_logits[0]

    answerParamPair = get_pos_with_logit(answerStartLogits, answerEndLogits)
    answer = Answer(tokenizer, inputs, outputs, answerParamPair)

    return answer


def get_answer(tokenizer, model, question, texts):
    """
    处理1个问题对应多篇文本，将从若干文本中分别抽取答案，同时为每份答案赋予分数（实际是概率），取最高者作为答案
    :param tokenizer: 用于将字符与token映射
    :param model: 模型
    :param question: 问题
    :param texts: 待抽取的文本集合（必须是可迭代对象）
    :return: 问题对应的答案和对应抽取的文本，如果没有答案，那么返回[CLS]
    """

    maxScore = 0
    answer = ""
    textForAnswer = ""
    for text in texts:
        possibleAnswer = get_possible_answer(tokenizer, model, question, text)
        if possibleAnswer is None:  # 如果未按预期得到答案，则跳过本轮
            continue
        score = possibleAnswer.get_score()
        if score > maxScore:
            maxScore = score
            answer = possibleAnswer.to_string()
            textForAnswer = text

    return answer, textForAnswer


if __name__ == '__main__':
    question = "银杏的寿命有多长"
    texts = [
        r"""
        银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等，其裸露的种子称为白果，叶称蒲扇。属裸子植物银杏门唯一现存物种，和它同门的所有其他物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食物。
        """,
        r"""
        银杏种子可以食用，在中国被称为白果，白果去壳后可煮熟直接食用，制作糖水配料等。以中医角度来说，据《本草纲目》记载：“白果小苦微甘，性温有小毒，多食令人腹胀”；银杏的果实内含小量氢氰酸毒素，性温，多食令人腹胀，遇热毒性减少，所以生吃或大量进食易引起中毒；多见于小儿；有呕吐、精神萎靡、发热、抽搐等征状。又说“熟食温肺、益气、定喘嗽、缩小便，止白浊，生食降痰，消毒杀虫，嚼浆涂鼻面手足，去鼻疽疱黑干黯皴皱及疥癣疳虫阴虱”。而银杏的种籽，即果仁有暖肺、止喘嗽及减少痰量之功效。特别是对于哮喘、慢性气管及支气管炎及肺结核等病症有明显的疗效。而且对补助泌尿系统有好处，滋阴益肾，可改善尿频。
        """,
    ]

    tokenizer, model = preload()
    print(get_answer(tokenizer, model, question, texts))
