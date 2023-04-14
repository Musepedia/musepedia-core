import grpc
import pytest


from Config import GRPC_PORT
from src.rpc.proto.GPT_pb2 import Exhibit
from src.rpc.proto import GPT_pb2, GPT_pb2_grpc


@pytest.fixture
def stub():
    channel = grpc.insecure_channel('localhost:{0}'.format(GRPC_PORT))
    stub = GPT_pb2_grpc.GPTServiceStub(channel)

    return stub


def test_get_answer_with_gpt(stub):
    user_question = '龙虎人丹是什么'
    exhibits = [Exhibit(label='龙虎人丹',
                        descriptions=['1907年，上海总商会通电全国，号召开展抵制日货的反日爱国运动，全国各地商界可谓是一呼百应。然而，只有日本产的仁丹仍然盘踞在中国城乡，到处可见翘着小胡子的仁丹招贴画。黄楚九对这小小仁丹在中国大地横行无忌愤愤不平。1909年黄楚九得到一张"诸葛行军散"的古方，同时参考自己祖传的《七十二症方》，反复研制出新的方剂，做成小粒药丸，取名为"人丹"。龙虎人丹上市后，黄楚九大力宣传，凡是贴着仁丹广告的地方，都贴上醒目的龙虎人丹广告，与其展开竞销。日本东亚公司眼看龙虎人丹对他们的仁丹构成了威胁，便控告人丹是"冒牌"、"侵权"，要求中国政府勒令停产。黄楚九聘请上海著名大律师，与日商大打官司。直至上诉到北京最高法院机关，1927年做出终审裁决，判定"人丹"与"仁丹"两药各不相干，可以同时在市场上销售。十年诉讼虽令黄楚九损失十万余元，但"人丹"的名声逐渐扩大，销路大增。史家评价:这场官司的胜诉，"给中国人出了一口气!'])]
    museum_name = '长风商标海报收藏馆'

    request = GPT_pb2.GPTRequest(user_question=user_question,
                                 exhibits=exhibits,
                                 museum_name=museum_name)

    response = stub.GetAnswerWithGPT(request)
    print(response.completion)

    assert response.completion is not None

