# -*- coding:utf-8 -*-

from elasticsearch import Elasticsearch
from loguru import logger
from Config import ELASTIC_SEARCH_HOST, ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD


class ESTools:
    """
    调用ElasticSearch的一些工具
    """

    def __init__(self, host=ELASTIC_SEARCH_HOST, username=ELASTIC_SEARCH_USERNAME, password=ELASTIC_SEARCH_PASSWORD):
        self.index_name = "paragraphs"
        self.es = Elasticsearch(
            hosts=host,
            basic_auth=(username, password)
        )
        if not self.es.ping():
            raise ValueError('ElasticSearch连接失败，原因：{0}'.format(self.es.info()))

    def has_key(self, key: str) -> bool:
        """
        判断关键字key是否存在

        :param key：关键字
        :return: key是否存在
        """
        body = {
            'term': {
                "name": key
            }
        }
        queryResult = self.es.search(index=self.index_name, query=body, size=1)["hits"]["max_score"]
        if queryResult is None:
            return False
        else:
            return True

    def create_index(self):
        """
        创建索引
        """

        _index_map = {
            "properties": {
                "name": {
                    "type": "keyword"
                },
                "title": {
                    "type": "text",
                    "analyzer": "ik_smart"
                },
                "content": {
                    "type": "text",
                    "analyzer": "ik_smart"
                },
                "source": {
                    "type": "keyword"
                },
                "valid": {
                    "type": "long"
                }
            }

        }
        if not self.es.indices.exists(index=self.index_name):
            result = self.es.indices.create(index=self.index_name, mappings=_index_map)
            if result.get("acknowledged"):
                logger.info("索引创建成功")
            else:
                logger.info(f"索引创建失败:{result}")
        else:
            logger.info("索引已存在无需重复创建!")

    def delete_index(self):
        """
        删除索引
        :return:null
        """
        self.es.indices.delete(index=self.index_name)

    def create_document(self, name: str, title: str, content: str, _id: int, source='wiki', valid=0):
        """
        在paragraphs下新建文档

        :param valid: 是否检查过
        :param name:（展品）名字
        :param title:（展品）当前文档的标题——大致内容
        :param content:（展品）文档内容
        :param _id: 文档id
        :param source: 文档来源
        """

        doc = {'name': name, 'title': title, 'content': content, 'source': source, 'valid': valid}
        try:
            res = self.es.create(index=self.index_name, document=doc, id=str(_id))
            logger.info(res)
        except:
            logger.warning("文档已存在！")

    def get_document_count(self) -> int:
        """
        获取索引下文档总数

        :return:文档总数
        """
        body = {'match_all': {}}
        return self.es.count(index=self.index_name, query=body)['count']

    def do_search(self, body: dict, k: int):
        """
        根据关键词返回文档

        :param body: 查询条件
        :param k: 返回的最大结果数
        :return:符合关键词的文档
        """
        return self.es.search(index=self.index_name, query=body, size=k)["hits"]["hits"]

    def update_doc(self, body: dict, _id):
        """
        更新文档内容

        :param body: 文档内容
        :param _id: 文档id
        :return: 更新结果
        """
        try:
            re = self.es.update(index=self.index_name, id=_id, body=body)
            logger.info(re)
        except:
            logger.warning("文档更新失败！")


if __name__ == "__main__":
    es = ESTools()
    es.delete_index()
    es.create_index()
