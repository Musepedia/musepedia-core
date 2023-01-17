from loguru import logger


def model_logging(model_name: str):
    """
    打印模型加载成功的日志信息

    :param model_name: 加载的模型名称
    """

    def _model_logging(run):
        def wrapper(*args, **kwargs):
            result = run(*args, **kwargs)
            logger.info('{0}加载成功'.format(model_name))
            return result
        return wrapper
    return _model_logging
