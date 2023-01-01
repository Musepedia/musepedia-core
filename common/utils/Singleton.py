def singleton(cls):
    """
    创建一个单例

    :param cls: 被装饰的类，该类将成为单例
    """

    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = cls(*args, **kwargs)

        return wrapper.instance

    wrapper.instance = None
    return wrapper


if __name__ == '__main__':
    @singleton
    class Singleton:
        def __init__(self, value):
            self.value = value

    class_a = Singleton(1)
    class_b = Singleton(2)
    print(class_a == class_b)
    print(class_a.value == class_b.value)
