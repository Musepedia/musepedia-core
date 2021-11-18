# -*- coding: UTF-8 -*-
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

MODEL_PATH = 'src/qa/models/roberta-base-chinese-extractive-qa'


def preload():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

    return tokenizer, model


def get_answer(tokenizer, model, question, text):
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
    inputIds = torch.tensor(inputs['input_ids'].tolist()[0])
    outputs = model(**inputs)

    answerStartScores = outputs.start_logits
    answerEndScores = outputs.end_logits

    answerStartPos = torch.argmax(answerStartScores)
    answerEndPos = torch.argmax(answerEndScores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputIds[answerStartPos:answerEndPos]))
    return answer.replace(' ', '')
