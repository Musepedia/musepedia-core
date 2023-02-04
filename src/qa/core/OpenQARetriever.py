# -*- coding: UTF-8 -*-
from elasticsearch import Elasticsearch

import Config


class OpenQARetriever:
    """
    开放域问答（Open domain QA）的检索器，用于从大量文本中找出最可能包含问题对应回答的文本
    """

    def __init__(self, host=Config.ELASTIC_SEARCH_HOST, username=Config.ELASTIC_SEARCH_USERNAME,
                 password=Config.ELASTIC_SEARCH_PASSWORD):
        self.index_name = "paragraphs"
        self.es = Elasticsearch(
            hosts=host,
            basic_auth=(username, password)
        )

    @staticmethod
    def check_texts_with_titles(texts: [str], titles: [str]) -> bool:
        """
        检查texts与titles是否长度一致

        :param texts: 文本集合
        :param titles: 标题集合
        :return: texts与titles长度是否一致
        """

        return len(texts) == len(titles)

    def get_top_k_text(self, question: str, k: int) -> [str]:
        """
        从texts中找出最可能包含question对应回答的k个文本

        :param question: 问题
        :param k: top k
        :return: 最可能包含question对应回答的k个文本
        """
        # 查询语句
        body = {
            'bool': {
                "must": [
                    {
                        "match": {
                            "content": question
                        }
                    }
                ]
            }
        }
        re = list()
        for i in self.es.search(index=self.index_name, query=body, size=k)["hits"]["hits"]:
            re.append(i['_source']['content'])
        return re
