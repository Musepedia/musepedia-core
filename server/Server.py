# -*- coding: utf-8 -*-

import grpc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from concurrent import futures
from server.proto import QA_pb2_grpc, QA_pb2
from src.qa.core.QuestionAnswering import preload, get_answer


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(QA_pb2_grpc.MyServiceServicer):
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model = model

    def SayHello(self, request, context):
        result = get_answer(self._tokenizer, self._model, request.question, request.text)
        return QA_pb2.HelloReply(answer='%s' % result)


def serve(tokenizer, model):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    QA_pb2_grpc.add_MyServiceServicer_to_server(Greeter(tokenizer, model), server)
    server.add_insecure_port('[::]:5555')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    tokenizer, model = preload()
    serve(tokenizer, model)

