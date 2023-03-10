import grpc
import pytest


from Config import GRPC_PORT
from src.rpc.proto import GPT_pb2, GPT_pb2_grpc


@pytest.fixture
def stub():
    channel = grpc.insecure_channel('localhost:{0}'.format(GRPC_PORT))
    stub = GPT_pb2_grpc.GPTServiceStub(channel)

    return stub


def test_get_answer_with_gpt(stub):
    context = GPT_pb2.GPTContext()
    context.user_question = '银杏'
    context.exhibit_label = '银杏'
    context.exhibit_description = '银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等[4]，其裸露的种子称为白果，叶称蒲扇[5]。'

    museum = GPT_pb2.Museum()
    museum.museum_name = '上海自然博物馆'
    museum.museum_description = '上海自然博物馆以“自然·人·和谐”为主题，通过“演化的乐章”、“生命的画卷”、“文明的史诗”三大主线，呈现了起源之谜、生命长河、演化之道、大地探珍、缤纷生命、生态万象、生存智慧、人地之缘、上海故事、未来之路等10个常设展区及临展厅、4D影院、探索中心等配套功能区域。'

    request = GPT_pb2.GPTRequest(context=context,
                                 museum=museum)

    response = stub.GetAnswerWithGPT(request)
    print(response.completion)

    assert response.completion is not None

