from torch.utils.data import Dataset


class EvaluationText:
    def __init__(self, text):
        self.text = text
        self.id = 0  # 仅用于占位


def qa_reader_dataset_wrapper(batch):
    return {'question': [batch[i]['question'] for i in range(len(batch))],
            'text': [[EvaluationText(text.text) for text in batch[i]['text']] for i in range(len(batch))],
            'id': [batch[i]['id'] for i in range(len(batch))],
            'answer': [batch[i]['answer'] for i in range(len(batch))],
            'answer_pos': ([batch[i]['answer_start'] for i in range(len(batch))],
                           [batch[i]['answer_start'] + len(batch[i]['answer']) for i in range(len(batch))])}


class QAReaderDataset(Dataset):
    def __init__(self, data_util):
        super(QAReaderDataset).__init__()
        self._data_util = data_util
        self._question_list = self._data_util.get_question()
        self._text_list = self._data_util.get_text()
        self._id_list = self._data_util.get_id()
        self._answer_list = self._data_util.get_answer()
        self._answer_start_list = self._data_util.get_answer_start()

    def __len__(self):
        return len(self._question_list)

    def __getitem__(self, idx):
        preprocess_text_list = []
        if not self._data_util.check_question_text_length(self._question_list[idx], self._text_list[idx]):
            valid_text_list = self._data_util.cut_text(self._text_list[idx])
            # todo 处理answer_start随着文本被切割，其值也应该发生相应的变化
            preprocess_text_list = [EvaluationText(valid_text) for valid_text in valid_text_list]
        else:
            preprocess_text_list.append(EvaluationText(self._text_list[idx]))

        return {'question': self._question_list[idx],
                'text': preprocess_text_list,
                'id': self._id_list[idx],
                'answer': self._answer_list[idx],
                'answer_start': self._answer_start_list[idx]}
