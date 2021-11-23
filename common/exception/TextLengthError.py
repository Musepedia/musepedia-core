class TextLengthError(BaseException):
    def __init__(self, length, max_length):
        self._length = length
        self._MAX_LENGTH = max_length

    def __str__(self):
        return format("输入的文章长度为 %d，文章长度不得大于 %d" % (self._length, self._MAX_LENGTH))
