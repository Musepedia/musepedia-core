import psutil
import os


class MemoryWarning(Warning):
    def __init__(self):
        self._memory_percentage = None

    @staticmethod
    def _check_memory():
        currentPid = os.getpid()
        currentProcess = psutil.Process(currentPid)
        memoryUsed = currentProcess.memory_info().rss / 1024 / 1024 / 1024

        return memoryUsed

    @staticmethod
    def _get_total_memory():
        totalMemory = psutil.virtual_memory().total / 1024 / 1024 / 1024

        return totalMemory

    def get_memory_percentage(self):
        """
        计算当前Python进程所占用的内存及比例
        :return: 当前Python进程所占用的内存，设备总内存
        """
        memoryUsed = MemoryWarning._check_memory()
        totalMemory = MemoryWarning._get_total_memory()

        self._memory_percentage = memoryUsed * 100 / totalMemory

        return memoryUsed, totalMemory

    def __str__(self):
        memoryUsed = self.get_memory_percentage()[0]

        return format("目前内存占用了 %f G，占总内存的 %f %%" % (memoryUsed, self._memory_percentage))
