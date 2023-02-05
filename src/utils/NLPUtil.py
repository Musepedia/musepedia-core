import jieba.analyse
import re


class NLPUtil:
    """
    自软语言处理工具类
    """

    # 不包含中英文标点符号与数字的正则表达式
    PUNCTUATION_DIGIT_PATTERN = r'''[^\u3002\uff1f\uff01\uff0c\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09
    \u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u300c\u300d\ufe43\ufe44\u3014\u3015\u2026\u2014\uff5e\ufe4f
    \uffe50-9%&',;=\[\]!?.$<>\x22]+'''

    @classmethod
    def get_keyword(cls, text: str) -> [str]:
        """
        基于TF-IDF算法获取关键词，同时对文本做全分词，结果取两个方法的并集，去除包含特殊字符的关键词（中英标点符号、数字等）

        :param text: 文本
        :return: top k关键词
        """

        tfidf_keyword = set(jieba.analyse.extract_tags(text, topK=len(text)//6))
        segmentation_keyword = set(jieba.lcut(text, cut_all=True))
        keyword = list(tfidf_keyword | segmentation_keyword)

        pattern = re.compile(cls.PUNCTUATION_DIGIT_PATTERN)
        keyword = [word for word in keyword if pattern.fullmatch(word)]

        return keyword

    @classmethod
    def get_exhibit_alias(cls, texts: [str]) -> [str]:
        keyword = set()
        for text in texts:
            keyword = keyword.union(set(jieba.analyse.extract_tags(text, topK=len(text) // 6)))

        keyword = list(keyword)
        pattern = re.compile(cls.PUNCTUATION_DIGIT_PATTERN)
        keyword = [word for word in keyword if pattern.fullmatch(word)]

        return keyword


if __name__ == '__main__':
    test_text = '银杏，落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等，其裸露的种子称为白果，叶称蒲扇。属裸子植物银杏门惟一现存物种，和它同门的所有其他物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食物。'

    keywords = NLPUtil.get_keyword(test_text)
    aliases = NLPUtil.get_exhibit_alias([test_text])
    print(keywords, len(keywords))
    print(aliases, len(keywords))
