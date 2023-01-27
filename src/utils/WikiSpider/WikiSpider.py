# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch

from langconv import *


def hk2s(context: str) -> str:
    """
    将繁体装换为简体

    :param context: 繁体文章
    :return:
    """
    return Converter("zh-hans").convert(context)


def context_process(context) -> str:
    """
    处理文章中的符号

    :param context:
    :return:返回处理好的文章
    """
    result = str()
    # 处理换行符
    for i in context:
        if i == ",":
            i = "，"
        if i == "\n":
            continue
        else:
            result += i
    context = result
    result = ""

    # 处理角标
    f = False
    for i in context:
        if not f:
            if i == "[":
                f = True
            else:
                result += i
        if f:
            if i == "]":
                f = False

    # 处理（）
    context = result
    result = ""
    f = False
    for i in context:
        if not f:
            if i == "（":
                f = True
            else:
                result += i
        if f:
            if i == "）":
                f = False
    # 处理()
    context = result
    result = ""
    f = False
    for i in context:
        if not f:
            if i == "(":
                f = True
            else:
                result += i
        if f:
            if i == ")":
                f = False
    return result


def divide_paragraph(paragraph: dict) -> dict:
    """
    划分文章段落

    :param paragraph: 初步分段的文章
    :return: 基于480长度对段落进行分割后的分段文章
    """
    para = {}
    _title = ""  # 新标题
    _sentence = ""  # 新段落
    flag = False
    for title in paragraph:
        if len(paragraph[title]) == 0:
            continue
        for sentence in paragraph[title].split('。'):
            if len(_sentence + sentence) < 480:
                _sentence += (sentence + '。')
            else:
                flag = True
                _title += (title + '_')
                if _title in para:
                    _title += '1'
                para[_title] = _sentence
                _sentence = (sentence + '。')
                _title = ''
        if flag:
            _title += (title + '_')
            flag = False
    return para


class WikiSpider:
    """
    Wikipedia爬虫，从Wikipedia上爬取关键词对应的文章内容
    """

    def __init__(self, host="http://pt.musepedia.cn:9200", username="elastic", password="r2tiq2FqAd5"):
        self.index_name = "paragraphs"
        self.es = Elasticsearch(
            hosts=host,
            basic_auth=(username, password)
        )
        self.header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/106.0.0.0 Safari/537.36 "
        }

    def hasKey(self, key: str) -> bool:
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
        queryResult = self.es.search(index=self.index_name, query=body, size=1)["hits"]
        if queryResult is None:
            return False
        else:
            return True

    def createIndex(self):
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
                }
            }

        }
        if not self.es.indices.exists(index=self.index_name):
            result = self.es.indices.create(index=self.index_name, mappings=_index_map)
            if result.get("acknowledged"):
                print("索引创建成功")
            else:
                print(f"索引创建失败:{result}")
        else:
            print("索引已存在无需重复创建!")

    def createDocument(self, name: str, title: str, content: str, source='wiki'):
        """
        在paragraphs下新建文档

        :param name:（展品）名字
        :param title:（展品）当前文档的标题——大致内容
        :param content:（展品）文档内容
        :param source: 文档来源
        """

        doc = {'name': name, 'title': title, 'content': content, 'source': source}
        try:
            self.es.create(index=self.index_name, id=name + '_' + title, document=doc)
        except:
            print("document exists.")

    def call_spider(self, keys: list[str]):
        """
        根据关键词在Wikipedia上进行对应文章的爬虫

        :param keys: 关键词的列表
        """
        # print(keys)
        for key in keys:
            print(key)
            responses = requests.get(
                "https://zh.wikipedia.org/wiki/" + key,
                headers=self.header,
                timeout=100
            )
            html = responses.text
            soup = BeautifulSoup(html, "lxml")
            pageinfo = ((soup.body.find(id="content").find(id="bodyContent")
                         .find(id="mw-content-text"))
            ).find(class_="mw-parser-output")

            # 获取目录
            cata = pageinfo.find(id="toc").select("li")
            catalogue = []
            for i in cata:
                if i.attrs['class'][0] == 'toclevel-1':
                    s = i.text.split('\n')[0]
                    catalogue.append(s[2:len(s)])

            # 主要处理部分
            context = pageinfo.find_all(['p', 'h2'])
            paragraph = {"概述": ""}
            title = "概述"
            f = True
            for one_sentence in context:
                m = one_sentence.text
                if f:
                    paragraph[title] = hk2s(m)
                    f = False
                if m[0:len(m) - 4] in catalogue:
                    title = hk2s(m[0:len(m) - 4])
                    paragraph[title] = ""
                else:
                    paragraph[title] += hk2s(m)
            # 文章符号处理
            for _title in paragraph:
                paragraph[_title] = context_process(paragraph[_title])
            # 文章长度处理
            paragraph = divide_paragraph(paragraph)
            # 导入es
            for _title in paragraph:
                self.createDocument(name=key, title=_title, content=paragraph[_title])

    def get_keys_1_recursive(self, original_key: str) -> [str]:
        """
        查询wikipedia上该original_key资料所有有超链接的关键词

        :param original_key: 初始关键字
        :return: 当前页面所有关键字列表
        """
        response = requests.get(
            "https://zh.wikipedia.org/wiki/" + original_key,
            headers=self.header,
            timeout=100
        )
        html = response.text
        soup = BeautifulSoup(html, "lxml")

        pageinfo = (
            (soup.body.find(id="content").find(id="bodyContent").find(id="mw-content-text"))
        ).find(class_="mw-parser-output")

        keys = {}
        if not self.hasKey(original_key):
            keys.append(original_key)
        tt = pageinfo.find_all('p')
        for j in tt:
            tt = j.find_all('a')
            for i in tt:
                if i['href'][0:7] == '/wiki/%':
                    key = Converter("zh-hans").convert(i.text)
                    if not es.hasKey(key):
                        keys.append(key)
                    else:
                        continue
        return keys
