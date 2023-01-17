import grpc


from Config import GRPC_PORT
from src.rpc.proto import QA_pb2, QA_pb2_grpc


def test_client_connection():
    channel = grpc.insecure_channel('localhost:{0}'.format(GRPC_PORT))
    stub = QA_pb2_grpc.MyServiceStub(channel)

    test_text = QA_pb2.RpcExhibitText()
    test_text.id = 1
    test_text.text = '银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。'

    request = QA_pb2.HelloRequest(question='银杏的寿命有多长',
                                  texts=[test_text],
                                  status=1)
    response = stub.SayHello(request)
    assert response.answerWithTextId.answer == '3000年'
