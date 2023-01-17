# -*- coding: utf-8 -*-
from loguru import logger

import psutil
import os


def concat_logging_info(pid, memory_used, memory_used_percentage):
    return "进程 {0} 使用了 {1:.3f}G 内存 ({2:.3f}%)".format(pid, memory_used, memory_used_percentage)


def os_logging(run):
    """
    打印系统信息，包括进程和当前占用内存
    """

    def _os_logging(*args, **kwargs):
        current_pid = os.getpid()
        current_process = psutil.Process(current_pid)
        memory_used = current_process.memory_info().rss / 1024 / 1024 / 1024
        memory_total = psutil.virtual_memory().total / 1024 / 1024 / 1024
        memory_used_percentage = memory_used / memory_total

        logger.warning(concat_logging_info(current_pid, memory_used, memory_used_percentage))
        result = run(*args, **kwargs)

        return result
    return _os_logging
