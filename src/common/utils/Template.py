import functools
import inspect

from src.qa.utils.TemplateUtil import TemplateUtil


def template(dict_name='variables'):
    """
    将被装饰的函数参数映射为dict，并存储至 ``dict_name`` 为参数名的dict中，作为关键字参数传入被装饰的函数

    :param dict_name: 作为关键字参数的参数名

    .. code-block::

        @template()
        def map_args(arg1: str, arg2: int, arg3: float, variables=None):
            return variables

    >>> map_args('test', 100, 3.14)
    {'arg1': 'test', 'arg2': 100, 'arg3': 3.14}
    """

    def _template(run):
        @functools.wraps(run)
        def wrapper(*args, **kwargs):
            variables = {}
            sig = inspect.signature(run)
            bound = sig.bind(*args, **kwargs).arguments
            variables.update({k: v for k, v in bound.items()})
            keyword_param = {dict_name: variables}

            return run(*args, **kwargs, **keyword_param)
        return wrapper
    return _template


if __name__ == '__main__':
    template_util = TemplateUtil('../../qa/templates/')

    @template()
    def map_args(arg1: str, arg2: int, arg3: float, variables=None):
        return variables

    print(map_args('test', 100, 3.14))
