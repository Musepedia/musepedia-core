### MGS-core

### common.log

- 日志模块，判断GRPC服务是否启动，是否结束
- 使用decorator封装日志输出部分
- 在server/Server.py文件的server函数外使用@service_start，即可输出日志

