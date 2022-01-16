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
        self._posWithLogitPair = pos_with_logit
        self._startScores = outputs.start_logits[0].detach().numpy()
        self._endScores = outputs.end_logits[0].detach().numpy()

    def get_pos(self):
        """
        返回答案的起始位置和终止位置，特别地，终止位置需要+1
        :return: 一个包含起始位置和终止位置的tuple
        """

        return self._posWithLogitPair[0][0], self._posWithLogitPair[1][0]+1

    def get_logits(self):
        """
        返回答案起始位置和终止位置对应的logits，这个结果可以被用来计算答案的得分
        :return: 一个包含起始位置和终止位置logits的tuple
        """

        return self._posWithLogitPair[0][1], self._posWithLogitPair[1][1]

    def _get_p_mask(self):
        numOfSpans = len(self._inputs['input_ids'])
        mask = np.asarray(
            [
                [tok != 1 for tok in self._inputs.sequence_ids(spanId)]
                for spanId in range(numOfSpans)
            ]
        )

        return mask[0].tolist()

    def _get_attention_mask(self):
        return self._inputs['attention_mask'][0].detach().numpy()

    @staticmethod
    def _get_mask(raw_p_mask, raw_attention_mask):
        tokens = np.abs(np.array(raw_p_mask) - 1) & raw_attention_mask
        undesiredTokensMask = tokens == 0.0

        return undesiredTokensMask

    def _get_start_end_score(self):
        rawMask = self._get_p_mask()
        rawAttentionMask = self._get_attention_mask()
        undesiredTokensMask = self._get_mask(rawMask, rawAttentionMask)

        startScore = np.where(undesiredTokensMask, -10000.0, self._startScores)
        endScore = np.where(undesiredTokensMask, -10000.0, self._endScores)

        startScore = np.exp(startScore - np.log(np.sum(np.exp(startScore), axis=-1, keepdims=True)))
        endScore = np.exp(endScore - np.log(np.sum(np.exp(endScore), axis=-1, keepdims=True)))

        return startScore, endScore

    def get_score(self):
        """
        计算这个答案的评估得分（这个得分实际是答案正确的概率，由Softmax算得）
        :return: 返回评估得分
        """

        start, end = self._get_start_end_score()
        answerPosition = self.get_pos()
        answerLength = answerPosition[1] - answerPosition[0]

        if start.ndim == 1:
            start = start[None]
        if end.ndim == 1:
            end = end[None]

        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))
        candidates = np.tril(np.triu(outer), answerLength-1)

        return np.max(candidates)

    def to_string(self):
        """
        将答案转为字符串表示
        :param tokenizer: 用于将字符与token映射
        :return: 返回字符串形式的答案
        """

        answerStartPos, answerEndPos = self.get_pos()
        inputIds = self._inputs['input_ids'][0]

        answer = self._tokenizer.convert_tokens_to_string(self._tokenizer.convert_ids_to_tokens(inputIds[answerStartPos:answerEndPos]))

        return answer.replace(' ', '')
