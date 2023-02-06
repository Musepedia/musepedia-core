# -*- coding: utf-8 -*-

import grpc


from src.common.exception.ExceptionHandler import catch
from src.common.log.ServiceLogging import service_logging
from concurrent import futures
from Config import GRPC_PORT, ROBERTA_MODEL_PATH
from src.qa.core.OpenQARetriever import OpenQARetriever
from src.rpc.proto import QA_pb2_grpc, QA_pb2
from src.qa.core.QuestionAnswering import QAReader
from src.qa.utils.MapUtil import MapUtil
from src.utils.NLPUtil import NLPUtil

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(QA_pb2_grpc.MyServiceServicer):
    def __init__(self):
        self._qa_reader = QAReader(model_path=ROBERTA_MODEL_PATH)
        self._map_util = MapUtil()
        self._open_qa_retriever = OpenQARetriever()
        self._nlp_util = NLPUtil()

    def GetAnswer(self, request: QA_pb2.QARequest, context):
        answer_with_text_id = QA_pb2.AnswerWithTextId()
        answer, text_id = self._qa_reader.get_answer(request.question, request.texts)
        if request.status == 2:
            answer = self._map_util.render_map(answer)
        if len(answer) == 0:
            # 没有答案，尝试OpenQA获取新的答案
            open_documents = self._open_qa_retriever.get_top_k_text(request.question, 15)
            answer = self._qa_reader.get_answer(request.question, open_documents)  # todo 需要open document的id
        answer_with_text_id.answer = answer
        answer_with_text_id.textId = text_id

        return QA_pb2.QAReply(answerWithTextId=answer_with_text_id)

    def GetOpenDocument(self, request: QA_pb2.OpenDocumentRequest, context):
        pass

    def GetExhibitAlias(self, request: QA_pb2.ExhibitLabelAliasRequest, context):
        alias_list = self._nlp_util.get_exhibit_alias(request.texts)

        return QA_pb2.ExhibitLabelAliasReply(aliases=alias_list)


@service_logging
@catch(KeyboardInterrupt)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    QA_pb2_grpc.add_MyServiceServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:{0}'.format(GRPC_PORT))
    server.start()
    server.wait_for_termination()
