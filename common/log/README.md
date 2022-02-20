### MGS-core

### common.log

- 日志模块，判断GRPC服务是否启动，是否结束
- 使用decorator封装日志输出部分
- 在server/Server.py文件的server函数外使用@service_start，即可输出日志
- 使用@qa_logging可以输出问答的详细信息，包括问题、抽取文本、答案和得分，需要传入问题和文本在被包装函数的参数位置
- 使用@os_logging在日志中输出当前Python进程占用的内存情况
