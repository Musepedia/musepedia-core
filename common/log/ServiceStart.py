from loguru import logger


def service_start(func):
    """
    grpc服务启动时记录日志
    """

    def _service_start(*args, **kwargs):
        logger.add('logs/server.log')
        logger.info('Grpc服务启动.')
        result = func(*args, **kwargs)
        logger.info('Grpc服务结束.')
        return result
    return _service_start
