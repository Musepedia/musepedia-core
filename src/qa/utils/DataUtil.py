import json
import time


class DataUtil:
    """
    用于处理data文件夹下的数据，可以按照给定格式解析json文件，对数据进行预处理等
    """

    MAX_LENGTH = 512
    MAX_TEXT_LENGTH = 470

    def __init__(self, data_path):
        self._data_path = data_path
        self._version, self._data = self.get_data_dict()

    def get_data_dict(self) -> (str, [dict]):
        """
        从文件地址中读取语料数据，文件必须是json格式，按照如下规则组织：

        {
            version: ...,
            data: [
                {
                    title: ...,
                    paragraphs: [
                        {
                            context: ...,
                            qas: [
                                {
                                    question: ...,
                                    answers: [
                                        {
                                            text: ...,
                                            answer_start: ...
                                        }
                                    ],
                                    id: ...
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        }

        :return: 返回语料的版本和语料数据构成的元组
        """

        with open(self._data_path, 'r') as file:
            data = json.load(file)

        return data['version'], data['data']

    def get_question(self) -> [str]:
        """
        返回问题列表

        :return: 问题列表
        """

        question_list = []
        for title_paragraphs_pair in self._data:
            for paragraph in title_paragraphs_pair['paragraphs']:
                for question_answer_pair in paragraph['qas']:
                    question_list.append(question_answer_pair['question'])

        return question_list

    def get_text(self) -> [str]:
        """
        返回待抽取文本列表，考虑到一篇文本可能有多个问题和回答，因此长度需要和问题列表对齐

        :return: 文本列表
        """

        text_list = []
        for title_paragraphs_pair in self._data:
            for paragraph in title_paragraphs_pair['paragraphs']:
                for _ in range(len(paragraph['qas'])):
                    text_list.append(paragraph['context'])

        return text_list

    def get_id(self) -> [str]:
        """
        返回问题和回答的id列表

        :return: id列表
        """

        id_list = []
        for title_paragraphs_pair in self._data:
            for paragraph in title_paragraphs_pair['paragraphs']:
                for question_answer_pair in paragraph['qas']:
                    id_list.append(question_answer_pair['id'])

        return id_list

    def get_answer(self) -> [str]:
        """
        返回回答列表

        :return: 回答列表
        """

        answer_list = []
        for title_paragraphs_pair in self._data:
            for paragraph in title_paragraphs_pair['paragraphs']:
                for question_answer_pair in paragraph['qas']:
                    for answer in question_answer_pair['answers']:
                        answer_list.append(answer['text'])

        return answer_list

    def get_answer_start(self) -> [str]:
        """
        返回回答起始位置列表

        :return: 回答起始位置列表
        """

        answer_start_list = []
        for title_paragraphs_pair in self._data:
            for paragraph in title_paragraphs_pair['paragraphs']:
                for question_answer_pair in paragraph['qas']:
                    for answer in question_answer_pair['answers']:
                        answer_start_list.append(answer['answer_start'])

        return answer_start_list

    @classmethod
    def check_question_text_length(cls, question: str, text: str) -> bool:
        return len(question) + len(text) < cls.MAX_LENGTH

    @classmethod
    def cut_text(cls, text: str) -> [str]:
        """
        针对超长文本（问题+文本的长度超过MAX_LENGTH），则需要对文本进行切割，以保证问题和文本长度能被模型接受

        :param text: 文本
        :return: 切割后的文本
        """

        split_text_list = []

        text_group = text.split('。')
        split_text = ''

        for i in range(len(text_group)):
            if i == len(text_group) - 1:
                if len(text_group[i]) + len(split_text) > cls.MAX_TEXT_LENGTH:
                    split_text_list.append(split_text)
                    split_text_list.append(text_group[i])
                else:
                    split_text += text_group[i]
                    split_text_list.append(split_text)
            else:
                text_group[i] += '。'
                if len(text_group[i]) + len(split_text) > cls.MAX_TEXT_LENGTH:
                    split_text_list.append(split_text)
                    split_text = text_group[i]
                else:
                    split_text += text_group[i]

        return split_text_list

    @staticmethod
    def get_current_time() -> str:
        """
        返回当前时间，格式为YYYYmmddHHMMSS

        :return: 当前时间的字符串
        """

        t = time.time()
        return time.strftime('%Y%m%d%H%M%S', time.localtime(t))
