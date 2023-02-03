# -*- coding: UTF-8 -*-
import evaluate
from torch.utils.data import DataLoader

from src.common.exception.ExceptionHandler import catch
from src.qa.core.QuestionAnswering import QAReader
from src.qa.utils.DataUtil import DataUtil
from src.qa.utils.QAReaderDataset import QAReaderDataset, qa_reader_dataset_wrapper


class EvaluationText:
    def __init__(self, text):
        self.text = text
        self.id = 0  # 仅用于占位


class QAReaderEvaluator:
    """
    用于加载QAReader，对模型性能进行评测
    """

    MODEL_NAME = 'roberta-base-chinese-extractive-qa'

    def __init__(self, model_path, test_data_path, batch_size, shuffle=False, num_workers=0, model_name=MODEL_NAME):
        self._model_path = model_path
        self._test_data_path = test_data_path
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._model_name = model_name
        self._reader = QAReader(self._model_path)
        self._reader.set_evaluation_mode()
        self._data_util = DataUtil(self._test_data_path)
        self._current_time = self._data_util.get_current_time()

    def get_dataloader(self):
        """
        由QAReaderDataset生成PyTorch的DataLoader

        :return: dataloader
        """

        reader_dataset = QAReaderDataset(self._data_util)

        return DataLoader(reader_dataset,
                          batch_size=self._batch_size,
                          shuffle=self._shuffle,
                          num_workers=self._num_workers,
                          collate_fn=qa_reader_dataset_wrapper)

    @catch(Exception)
    def evaluate(self):
        """
        评测模型的性能

        :return: 模型在测试集上的准确率和f1值
        """

        prediction_answer_list = []
        reference_answer_list = []

        dataloader = self.get_dataloader()

        for i, batch in enumerate(dataloader):
            predict_answer = self._reader.get_answer_batch(batch['question'], batch['text'],
                                                           batch_size=self._batch_size)

            for j in range(self._batch_size):
                prediction_answer_list.append(
                    {
                        'prediction_text': predict_answer[0][j],
                        'id': batch['id'][j]
                    }
                )
                reference_answer_list.append(
                    {
                        'answers': {
                            'answer_start': [0],  # 仅用于占位，实际评测不考虑答案的起始位置
                            'text': [batch['answer'][j]]
                        },
                        'id': batch['id'][j]
                    }
                )

        squad_metric = evaluate.load('squad')
        print(squad_metric.compute(predictions=prediction_answer_list,
                                   references=reference_answer_list))


if __name__ == '__main__':
    qa_reader_evaluator = QAReaderEvaluator('../models/roberta-base-chinese-extractive-qa',
                                            '../data/MLQA/test-context-zh-question-zh.json',
                                            2)

    qa_reader_evaluator.evaluate()
