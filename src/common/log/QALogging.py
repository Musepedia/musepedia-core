# -*- coding: utf-8 -*-
from loguru import logger

from src.common.log.BaseLogging import *


def concat_logging_info(question, text, answer, score):
    return "问题：{0}\t抽取文本：{1}\t答案：{2}\t得分：{3:.4f}".format(question, text, answer, score)


def qa_logging(question_index, text_index):
    """
    打印与QA相关的Debug级别日志信息，包括传入模型的问题、抽取的文本、答案和得分
    :param question_index: 问题在函数参数中的位置
    :param text_index: 抽取的文本在函数参数中的位置
    """

    def _qa_logging(run):
        def wrapper(*args, **kwargs):
            result = run(*args, **kwargs)
            question = args[question_index]
            text = args[text_index]
            if len(text) > MAX_TEXT_LENGTH_IN_LOG:
                text = text[:MAX_TEXT_LENGTH_IN_LOG] + '...'

            # result[0]仅适用于返回结果是tuple的情况
            logger.debug(concat_logging_info(question, text, result[0].to_string(), result[0].get_score()))
            return result
        return wrapper
    return _qa_logging


def qa_logging_batch(question_list_index, text_list_index):
    """
    打印与QA相关的Debug级别日志信息，包括传入模型的问题、抽取的文本、答案和得分
    :param question_list_index: 问题batch在函数参数中的位置
    :param text_list_index: 抽取的文本batch在函数参数中的位置
    """

    def _qa_logging(run):
        def wrapper(*args, **kwargs):
            result = run(*args, **kwargs)
            for i in range(len(args[question_list_index])):
                question = args[question_list_index][i]
                text = args[text_list_index][i]
                if len(text) > MAX_TEXT_LENGTH_IN_LOG:
                    text = text[:MAX_TEXT_LENGTH_IN_LOG] + '...'

                # result[0]仅适用于返回结果是tuple的情况
                logger.debug(concat_logging_info(question, text, result[0][i].to_string(), result[0][i].get_score()))
            return result
        return wrapper
    return _qa_logging


if __name__ == '__main__':
    pass
