# -*- coding: UTF-8 -*-

from src.utils.ESTools import ESTools


class OpenQARetriever:
    """
    开放域问答（Open domain QA）的检索器，用于从大量文本中找出最可能包含问题对应回答的文本
    """

    def __init__(self):
        pass

    @staticmethod
    def check_texts_with_titles(texts: [str], titles: [str]) -> bool:
        """
        检查texts与titles是否长度一致

        :param texts: 文本集合
        :param titles: 标题集合
        :return: texts与titles长度是否一致
        """

        return len(texts) == len(titles)

    def get_top_k_text(self, question: str, k: int, name=None) -> [(str, str)]:
        """
        从texts中找出最可能包含question对应回答的k个文本

        :param name: 限定的对象名字
        :param question: 问题
        :param k: top k
        :return: 最可能包含question对应回答的k个文本及其id
        """
        # 查询语句

        body = {
            'bool': {
                "must": [
                    {
                        "term": {
                            "valid": 1
                        }
                    },
                    {
                        "match": {
                            "content": question
                        }
                    }
                ]
            }
        }
        if name is not None:
            body['bool']['must'].append({
                "term": {
                    "name": name
                }
            })
        result = list()
        es = ESTools()
        for i in es.do_search(body=body, k=k):
            result.append((i['_source']['content'], int(i['_id'])))
        return result


if __name__ == '__main__':
    qa = OpenQARetriever()
    re = qa.get_top_k_text(question="女王凤凰螺的分布", k=10, name="女王凤凰螺")
    print(re)
    re = qa.get_top_k_text(question="女王凤凰螺的分布", k=10)
    print(re)
