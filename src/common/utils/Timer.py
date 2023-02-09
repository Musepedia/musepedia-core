import time


def timer(run):
    """
    用于记录一个函数的运行时长
    """

    def _timer(*args, **kwargs):
        start = time.time()
        result = run(*args, **kwargs)
        end = time.time()
        print('函数 {0}() 耗时 {1:.4} s'.format(run.__name__, end - start))
        return result
    return _timer
