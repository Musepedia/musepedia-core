# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.common.exception.ExceptionHandler import catch
from src.utils.WikiSpider.langconv import *
from src.utils.ESTools import ESTools


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

    def __init__(self):
        self.header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/106.0.0.0 Safari/537.36 "
        }
        self.es = ESTools()

    @catch(Exception)
    def call_spider(self, keys: [str]):
        """
        根据关键词在Wikipedia上进行对应文章的爬虫

        :param keys: 关键词的列表
        """
        for key in keys:
            logger.info(key)
            responses = requests.get(
                "https://zh.wikipedia.org/wiki/" + key,
                headers=self.header,
                timeout=100, 
                verify=False
            )
            html = responses.text
            soup = BeautifulSoup(html, "lxml")
            pageinfo = ((soup.body.find(id="content").find(id="bodyContent")
                         .find(id="mw-content-text"))).find(class_="mw-parser-output")

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
            _id = self.es.get_document_count()
            for _title in paragraph:
                _id += 1
                self.es.create_document(name=key, title=_title, content=paragraph[_title], _id=_id)

    @catch(Exception)
    def get_keys_1_recursive(self, original_key: str):
        """
        查询wikipedia上该original_key资料所有有超链接的关键词

        :param original_key: 初始关键字
        :return: 当前页面所有关键字列表
        """
        response = requests.get(
            "https://zh.wikipedia.org/wiki/" + original_key,
            headers=self.header,
            timeout=100, 
            verify=False
        )

        html = response.text
        soup = BeautifulSoup(html, "lxml")

        pageinfo = (
            (soup.body.find(id="content").find(id="bodyContent").find(id="mw-content-text"))
        ).find(class_="mw-parser-output")

        keys = list()
        if not self.es.has_key(original_key):
            keys.append(original_key)
        tt = pageinfo.find_all('p')
        for j in tt:
            tt = j.find_all('a')
            for i in tt:
                if i['href'][0:7] == '/wiki/%':
                    key = Converter("zh-hans").convert(i.text)
                    if not self.es.has_key(key):
                        keys.append(key)
                    else:
                        continue
        return keys


if __name__ == '__main__':
    # serve()
    spider = WikiSpider()
    spider.call_spider(["狼"])
