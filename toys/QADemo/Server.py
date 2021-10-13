# -*- coding: utf-8 -*-

import time

import grpc
import proto

from concurrent import futures
from proto import QA_pb2, QA_pb2_grpc
from toys.QADemo.QuestionAnswering import get_answer


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(proto.QA_pb2_grpc.MyServiceServicer):
    def SayHello(self, request, context):
        result = get_answer(request.question, request.text)
        return QA_pb2.HelloReply(answer='%s!' % result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    QA_pb2_grpc.add_MyServiceServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:5555')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()

