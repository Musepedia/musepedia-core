import time


def timer(run):
    """
    定义函数时添加 @timer 可以计算执行时间
    """

    def _timer(*args, **kwargs):
        start = time.perf_counter()
        res = run(*args, **kwargs)
        end = time.perf_counter()
        print("Function '{0}()' Elapsed time: {1} s".format(run.__name__, end - start))
        if res is not None:
            return res
    return _timer
