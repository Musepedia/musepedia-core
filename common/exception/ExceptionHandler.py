import traceback

from common.exception.TextLengthError import TextLengthError


def catch(*exception):
    """
    捕获异常
    :param exception: 异常类型，必须继承BaseException类
    """

    def _catch(run):
        def wrapper(*args, **kwargs):
            result = None
            try:
                result = run(*args, **kwargs)
            except exception:
                traceback.print_exc()
            if result is not None:
                return result
        return wrapper
    return _catch


def check_length(max_length=512):
    """
    检查函数传入的参数，如果是str类型的，长度不允许超过max_length
    主要用于检测输入模型的文章长度，确保不超过模型规定的长度上限
    :param max_length: 最大长度，默认512
    """

    def _check_length(run):
        def wrapper(*args, **kwargs):
            for argument in args:
                if type(argument) is str and len(argument) > max_length:  # str类型的变量长度不能超过上限max_length
                    raise TextLengthError(len(argument), max_length)
            result = run(*args, **kwargs)
            return result
        return wrapper
    return _check_length
