# -*- coding: UTF-8 -*-
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# 请在部署时使用绝对路径，相对路径可能会导致模型无法正确被加载
MODEL_PATH = 'models/roberta-base-chinese-extractive-qa'


def get_answer(question, text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
    inputIds = torch.tensor(inputs['input_ids'].tolist()[0])
    outputs = model(**inputs)

    answerStartScores = outputs.start_logits
    answerEndScores = outputs.end_logits

    answerStartPos = torch.argmax(answerStartScores)
    answerEndPos = torch.argmax(answerEndScores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputIds[answerStartPos:answerEndPos]))
    return answer.replace(' ', '')


if __name__ == '__main__':
    question = "银杏的寿命有多长"
    text = r"""银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等，其裸露的种子称为白果，叶称蒲扇。属裸子植物银杏门唯一现存物种，和它同门的所有其他物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食物。
银杏种子可以食用，在中国被称为白果，白果去壳后可煮熟直接食用，制作糖水配料等。
以中医角度来说，据《本草纲目》记载：“白果小苦微甘，性温有小毒，多食令人腹胀”；银杏的果实内含小量氢氰酸毒素，性温，多食令人腹胀，遇热毒性减少，所以生吃或大量进食易引起中毒；多见于小儿；有呕吐、精神萎靡、发热、抽搐等征状。又说“熟食温肺、益气、定喘嗽、缩小便，止白浊，生食降痰，消毒杀虫，嚼浆涂鼻面手足，去鼻疽疱黑干黯皴皱及疥癣疳虫阴虱”。而银杏的种籽，即果仁有暖肺、止喘嗽及减少痰量之功效。特别是对于哮喘、慢性气管及支气管炎及肺结核等病症有明显的疗效。而且对补助泌尿系统有好处，滋阴益肾，可改善尿频。
"""
    print(get_answer(question, text))
