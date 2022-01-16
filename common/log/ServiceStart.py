from loguru import logger


def service_start(func):
    """
    grpc服务启动时记录日志
    """

    def _service_start(*args, **kwargs):
        startHandlerId = logger.add('logs/server.log')
        logger.info('Grpc服务启动.')
        logger.remove(startHandlerId)

        result = func(*args, **kwargs)

        endHandlerId = logger.add('logs/server.log')
        logger.info('Grpc服务结束.')
        logger.remove(endHandlerId)
        return result
    return _service_start
