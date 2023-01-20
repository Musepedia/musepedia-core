import traceback

from loguru import logger


def model_logging(model_name: str):
    """
    打印模型加载成功的日志信息

    :param model_name: 加载的模型名称
    """

    def _model_logging(run):
        def wrapper(*args, **kwargs):
            try:
                result = run(*args, **kwargs)
            except OSError:
                logger.error('{0}加载失败'.format(model_name))
                logger.error(traceback.format_exc())
                return None
            logger.info('{0}加载成功'.format(model_name))
            return result
        return wrapper
    return _model_logging
