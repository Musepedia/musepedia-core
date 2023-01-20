import numpy as np


class Answer:
    def __init__(self, tokenizer, inputs, outputs, pos_with_logit):
        """
        :param tokenizer: tokenizer
        :param inputs: 经过tokenizer tokenize后的输入数据
        :param outputs: 经过模型计算得到的输出
        :param pos_with_logit: 包含答案起始位置与终止位置和logits的tuple
        """

        self._tokenizer = tokenizer
        self._inputs = inputs
        self._pos_with_logit_pair = pos_with_logit
        self._start_scores = outputs.start_logits[0].cpu().detach().numpy()
        self._end_scores = outputs.end_logits[0].cpu().detach().numpy()

    def get_pos(self):
        """
        返回答案的起始位置和终止位置，特别地，终止位置需要+1
        :return: 一个包含起始位置和终止位置的tuple
        """

        return self._pos_with_logit_pair[0][0], self._pos_with_logit_pair[1][0] + 1

    def get_logits(self):
        """
        返回答案起始位置和终止位置对应的logits，这个结果可以被用来计算答案的得分
        :return: 一个包含起始位置和终止位置logits的tuple
        """

        return self._pos_with_logit_pair[0][1], self._pos_with_logit_pair[1][1]

    def _get_p_mask(self):
        num_of_spans = len(self._inputs['input_ids'])
        mask = np.asarray(
            [
                [tok != 1 for tok in self._inputs.sequence_ids(span_id)]
                for span_id in range(num_of_spans)
            ]
        )

        return mask[0].tolist()

    def _get_attention_mask(self):
        return self._inputs['attention_mask'][0].cpu().detach().numpy()

    @staticmethod
    def _get_mask(raw_p_mask, raw_attention_mask):
        tokens = np.abs(np.array(raw_p_mask) - 1) & raw_attention_mask
        undesired_tokens_mask = tokens == 0.0

        return undesired_tokens_mask

    def _get_start_end_score(self):
        raw_mask = self._get_p_mask()
        raw_attention_mask = self._get_attention_mask()
        undesired_tokens_mask = self._get_mask(raw_mask, raw_attention_mask)

        start_score = np.where(undesired_tokens_mask, -10000.0, self._start_scores)
        end_score = np.where(undesired_tokens_mask, -10000.0, self._end_scores)

        start_score = np.exp(start_score - np.log(np.sum(np.exp(start_score), axis=-1, keepdims=True)))
        end_score = np.exp(end_score - np.log(np.sum(np.exp(end_score), axis=-1, keepdims=True)))

        return start_score, end_score

    def get_score(self):
        """
        计算这个答案的评估得分（这个得分实际是答案正确的概率，由Softmax算得）
        :return: 返回评估得分
        """

        start, end = self._get_start_end_score()
        answer_position = self.get_pos()
        answer_length = answer_position[1] - answer_position[0]

        if start.ndim == 1:
            start = start[None]
        if end.ndim == 1:
            end = end[None]

        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))
        candidates = np.tril(np.triu(outer), answer_length-1)

        return np.max(candidates)

    def to_string(self):
        """
        将答案转为字符串表示
        :param tokenizer: 用于将字符与token映射
        :return: 返回字符串形式的答案
        """

        answer_start_pos, answer_end_pos = self.get_pos()
        input_ids = self._inputs['input_ids'][0]

        answer = self._tokenizer.convert_tokens_to_string(self._tokenizer.convert_ids_to_tokens(input_ids[answer_start_pos:answer_end_pos]))

        return answer.replace(' ', '')
