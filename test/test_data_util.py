from src.qa.utils.DataUtil import DataUtil
from src.qa.utils.QAReaderDataset import QAReaderDataset


def test_length():
    data_util = DataUtil('src/qa/data/MLQA/test-context-zh-question-zh.json')
    questions = data_util.get_question()
    texts = data_util.get_text()
    ids = data_util.get_id()
    answers = data_util.get_answer()
    answer_starts = data_util.get_answer_start()
    assert len(questions) == len(texts) == len(ids) == len(answers) == len(answer_starts)


def test_cut_text():
    data_util = DataUtil('src/qa/data/MLQA/test-context-zh-question-zh.json')
    questions = data_util.get_question()
    texts = data_util.get_text()

    for i in range(len(questions)):
        if not DataUtil.check_question_text_length(questions[i], texts[i]):
            cut_text = data_util.cut_text(texts[i])

            # 1. 切割后的文本必须小于规定长度
            for text in cut_text:
                assert len(text) <= DataUtil.MAX_TEXT_LENGTH, '第{0}个测试用例断言失败'.format(i)

            # 2. 切割后的文本可以被还原成原来的文本
            assert texts[i] == ''.join(cut_text), '第{0}个测试用例断言失败'.format(i)
