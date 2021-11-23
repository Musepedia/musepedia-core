### MGS-core

### common.exception
- 异常处理模块，用装饰器封装异常处理部分
- `@catch(*exception)`可以传入异常类型并被捕获
- `@check_length(max_length)`可以检查传入文章长度是否超过`max_length`，现在用于检测文章长度是否超过模型能接受的长度上限，
这会在长度超过时抛出异常，请配合`@catch`捕获异常