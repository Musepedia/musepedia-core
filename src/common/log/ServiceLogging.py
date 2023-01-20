import torch.cuda
from loguru import logger

from Config import USE_GPU


def service_logging(func):
    """
    grpc服务启动时记录日志
    """

    def _service_logging(*args, **kwargs):
        logger.info('Grpc服务启动.')
        is_cuda_available = torch.cuda.is_available()
        logger.info('CUDA可用，模型可用GPU运行' if is_cuda_available else 'CUDA不可用，模型使用CPU运行')
        if is_cuda_available and USE_GPU:
            cuda_device_count = torch.cuda.device_count()
            logger.info('共有{0}张GPU可用'.format(cuda_device_count))
            for i in range(cuda_device_count):
                logger.info('device {0}: {1}'.format(i, torch.cuda.get_device_name(i)))
        result = func(*args, **kwargs)
        logger.info('Grpc服务结束.')
        return result
    return _service_logging
