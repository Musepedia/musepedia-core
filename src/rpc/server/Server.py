# -*- coding: utf-8 -*-

import grpc


from src.common.exception.ExceptionHandler import catch
from src.common.log.ServiceLogging import service_logging
from concurrent import futures
from Config import GRPC_PORT, ROBERTA_MODEL_PATH, TEMPLATE_PATH
from src.qa.core.GPT import GPT, GPTContext
from src.qa.core.OpenQARetriever import OpenQARetriever
from src.rpc.proto import QA_pb2_grpc, QA_pb2, GPT_pb2, GPT_pb2_grpc
from src.qa.core.QuestionAnswering import QAReader
from src.qa.utils.MapUtil import MapUtil
from src.utils.NLPUtil import NLPUtil
from src.utils.WikiSpider.WikiSpider import WikiSpider

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(QA_pb2_grpc.MyServiceServicer):
    def __init__(self):
        self._qa_reader = QAReader(model_path=ROBERTA_MODEL_PATH)
        self._map_util = MapUtil()
        self._open_qa_retriever = OpenQARetriever()
        self._nlp_util = NLPUtil()
        self._wiki_spider_util = WikiSpider()

    def GetAnswer(self, request: QA_pb2.QARequest, context):
        answer_with_text_id = QA_pb2.AnswerWithTextId()
        answer, text_id = self._qa_reader.get_answer(request.question, request.texts)
        if request.status == 2:
            answer = self._map_util.render_map(answer)
        if len(answer) == 0:
            # 没有答案，尝试OpenQA获取新的答案
            open_documents = self._open_qa_retriever.get_top_k_text(request.question, 15)
            answer, text_id = self._qa_reader.get_answer(request.question, open_documents)  # todo 需要open document的id
        answer_with_text_id.answer = answer
        answer_with_text_id.text_id = text_id

        return QA_pb2.QAReply(answer_with_text_id=answer_with_text_id)

    def GetOpenDocument(self, request: QA_pb2.OpenDocumentRequest, context):
        original_keys = []
        for text in request.texts:
            original_keys.append(text.text)
        for one_key in original_keys:
            self._wiki_spider_util.call_spider(self._wiki_spider_util.get_keys_1_recursive(one_key))

    def GetExhibitAlias(self, request: QA_pb2.ExhibitLabelAliasRequest, context):
        alias_list = self._nlp_util.get_exhibit_alias(request.texts)

        return QA_pb2.ExhibitLabelAliasReply(aliases=alias_list)


class GPTService(GPT_pb2_grpc.GPTServiceServicer):
    def __init__(self):
        self._gpt = GPT(template_dir_path=TEMPLATE_PATH)

    def GetAnswerWithGPT(self, request: GPT_pb2.GPTRequest, context):
        user_prompt = self._gpt.create_user_prompt(GPTContext(user_question=request.context.user_question,
                                                              exhibit_label=request.context.exhibit_label,
                                                              exhibit_description=request.context.exhibit_description))
        system_prompt = self._gpt.create_system_prompt(museum_name=request.museum.museum_name,
                                                       museum_description=request.museum.museum_description)
        gpt_completion = self._gpt.generate(user_prompt, system_prompt)
        if gpt_completion is not None:
            return GPT_pb2.GPTReply(prompt=gpt_completion.prompt,
                                    completion=gpt_completion.completion,
                                    prompt_tokens=gpt_completion.prompt_tokens,
                                    completion_tokens=gpt_completion.completion_tokens)

        return gpt_completion


@service_logging
@catch(KeyboardInterrupt)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    QA_pb2_grpc.add_MyServiceServicer_to_server(Greeter(), server)
    GPT_pb2_grpc.add_GPTServiceServicer_to_server(GPTService(), server)
    server.add_insecure_port('[::]:{0}'.format(GRPC_PORT))
    server.start()
    server.wait_for_termination()
