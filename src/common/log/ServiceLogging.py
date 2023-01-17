from loguru import logger


def service_logging(func):
    """
    grpc服务启动时记录日志
    """

    def _service_logging(*args, **kwargs):
        logger.info('Grpc服务启动.')
        result = func(*args, **kwargs)
        logger.info('Grpc服务结束.')
        return result
    return _service_logging
