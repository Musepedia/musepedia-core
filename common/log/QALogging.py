# -*- coding: utf-8 -*-
from loguru import logger


_MAX_TEXT_LENGTH_IN_LOG = 20  # 在日志中最大表示的文本长度，超过该长度将用 ... 代替


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
            qaHandlerId = logger.add('logs/server.log')

            result = run(*args, **kwargs)
            question = args[question_index]
            text = args[text_index]
            if len(text) > _MAX_TEXT_LENGTH_IN_LOG:
                text = text[:_MAX_TEXT_LENGTH_IN_LOG] + '...'

            logger.debug(concat_logging_info(question, text, result.to_string(), result.get_score()))
            logger.remove(qaHandlerId)
            return result
        return wrapper
    return _qa_logging


if __name__ == '__main__':
    pass
