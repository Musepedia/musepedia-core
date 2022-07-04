# -*- coding: utf-8 -*-

import grpc
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from common.exception.ExceptionHandler import catch
from common.log.BaseLogging import init_logger
from common.log.ServiceLogging import service_logging
from concurrent import futures
from server.proto import QA_pb2_grpc, QA_pb2
from src.qa.core.QuestionAnswering import preload, get_answer_parallel
from src.qa.utils.Map import render_map
from src.analysis.core.question_analysis import question_analysis
from src.analysis.core.user_question_analysis import user_question_analysis
from src.analysis.core.info_analysis import info_analysis


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(QA_pb2_grpc.MyServiceServicer):
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model = model

    def SayHello(self, request: QA_pb2.HelloRequest, context):
        answerWithTextId = QA_pb2.AnswerWithTextId()
        answer, textId = get_answer_parallel(self._tokenizer, self._model, request.question, request.texts)
        if request.status == 2:
            answer = render_map(answer)
        answerWithTextId.answer = answer
        answerWithTextId.textId = textId

        return QA_pb2.HelloReply(answerWithTextId=answerWithTextId)

    def QuestionAnalysis(self, request: QA_pb2.QuestionAnalysisRequest):
        questions, questionFreq, labels, labelFreq = question_analysis(request, 5)
        return QA_pb2.QuestionAnalysisReply(questions=questions, questionFreq=questionFreq, exhibitLabels=labels,
                                            labelFreq=labelFreq)

    def UserQuestionAnalysis(self, request: QA_pb2.UserAnalysisRequest):
        user_preference = user_question_analysis(request)
        return QA_pb2.QuestionAnalysisReply(exhibitLabels=user_preference)

    def InfoAnalysis(self, request: QA_pb2.InfoAnalysisRequest):
        return QA_pb2.InfoAnalysisReply(ageWithLabls=info_analysis(request, 5))

@service_logging
@catch(KeyboardInterrupt)
def serve(tokenizer, model):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    QA_pb2_grpc.add_MyServiceServicer_to_server(Greeter(tokenizer, model), server)
    server.add_insecure_port('[::]:5555')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    init_logger()
    tokenizer, model = preload()
    serve(tokenizer, model)
