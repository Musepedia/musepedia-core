import os.path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from src.qa.core.QuestionAnswering import QAReader
from src.qa.evaluation.QAReaderEvaluator import QAReaderEvaluator
from src.qa.utils.DataUtil import DataUtil
from src.qa.utils.QAReaderDataset import QAReaderDataset, qa_reader_dataset_wrapper


class QAReaderTrainer:
    MODEL_NAME = 'roberta-base-chinese-extractive-qa'

    def __init__(self, model_path, checkpoint_path, train_data_path, dev_data_path,
                 batch_size, epoch_size, learning_rate,
                 shuffle=True, num_workers=0, model_name=MODEL_NAME):
        self._model_path = model_path
        self._checkpoint_path = checkpoint_path
        self._train_data_path = train_data_path
        self._dev_data_path = dev_data_path
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._learning_rate = learning_rate
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._model_name = model_name
        self._reader = QAReader(self._model_path)
        self._reader.set_train_mode()
        self._data_util = DataUtil(self._dev_data_path)
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

    def get_optimizer(self):
        optimizer = AdamW(self._reader.model.parameters(), lr=self._learning_rate)

        return optimizer

    def get_scheduler(self, dataloader, optimizer):
        num_training_steps = self._epoch_size * len(dataloader)
        scheduler = get_scheduler('linear',
                                  optimizer=optimizer,
                                  num_warmup_steps=0,
                                  num_training_steps=num_training_steps)

        return scheduler

    def get_checkpoint_filename(self, index):
        """
        获取模型checkpoint的文件名称，格式为checkpoint_[model_name]_[current_time]_[index]

        :param index: 第index个epoch
        :return: checkpoint的文件名
        """

        return 'checkpoint_{0}_{1}_{2}'.format(self._model_name, self._current_time, str(index))

    def train(self):
        dataloader = self.get_dataloader()
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(dataloader, optimizer)

        progress_bar = tqdm(range(self._epoch_size * len(dataloader)), position=0, leave=True)

        for epoch in range(self._epoch_size):
            for i, batch in enumerate(dataloader):
                predict_answer = self._reader.get_answer_batch(batch['question'], batch['text'],
                                                               batch_size=self._batch_size,
                                                               answer=batch['answer_pos'])

                batch_loss = predict_answer[2]
                batch_loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)

            torch.save(self._reader.model.state_dict(),
                       '{0}/{1}'.format(self._checkpoint_path, self.get_checkpoint_filename(epoch)))


if __name__ == '__main__':
    qa_reader_trainer = QAReaderTrainer(model_path='../models/roberta-base-chinese-extractive-qa',
                                        checkpoint_path='../models/checkpoints',
                                        train_data_path='../data/MLQA/dev-context-zh-question-zh.json',
                                        dev_data_path='../data/MLQA/dev-context-zh-question-zh.json',
                                        batch_size=1,
                                        epoch_size=2,
                                        learning_rate=1e-05,
                                        shuffle=False)

    qa_reader_trainer.train()
