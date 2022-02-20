# -*- coding: utf-8 -*-
from loguru import logger


# 在日志中最大表示的文本长度，超过该长度的部分将用 ... 代替
MAX_TEXT_LENGTH_IN_LOG = 20


def init_logger():
    """
    初始化日志配置，整个项目中提供唯一一个logger
    日志命名为server_创建日志时间.log，每周会创建一个新的日志文件保存日志信息
    """

    logger.add('logs/server_{time}.log', rotation="1 week")
