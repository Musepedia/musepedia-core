# -*- coding: UTF-8 -*-
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('../models/roberta-base-chinese-extractive-qa')
model = AutoModelForQuestionAnswering.from_pretrained('../models/roberta-base-chinese-extractive-qa')

questionAnswering = pipeline('question-answering', model=model, tokenizer=tokenizer)

texts = [
    r"""
    榕亚属，是桑科榕属的亚属之一，由于多为常绿乔木及细节分冶不易被人分辨，因此也统称榕树。有“正榕”、“鸟榕”、“老公须”或“戏叶榕树”等别名。原产于中国、日本、印度、菲律宾、马来西亚。
榕树为常绿大乔木，气根多数；叶互生，倒卵形或椭圆形，表面深绿色；隐花果，无柄单生或对生于叶腋，成熟时呈黄褐色、红褐色或黑紫色。
北宋治平元年（1064年），太守张伯玉移知福州，夏天酷热难耐，遂令编户浚沟七尺，种植榕树，后来“绿荫满城，暑不张盖”，故福州又有“榕城”的美称。全世界现有1000多种榕树，多集中在热带雨林地区，热带植物中最大的木本树种之一，常高达20米。由于耐污性好、易栽植，也是常见的行道树。
榕树的拉丁文为banyan，在西方这个名字多指来自印度及孟加拉等地的孟加拉榕；但在中国及亚洲等地则多指细叶榕。由于它们同属榕亚属，并且有相近的生活周期及气根等特征，因此榕树现多作为桑科榕属下榕亚属的简称。
    """,
    r"""
    银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等，其裸露的种子称为白果，叶称蒲扇。属裸子植物银杏门唯一现存物种，和它同门的所有其他物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食物。
银杏种子可以食用，在中国被称为白果，白果去壳后可煮熟直接食用，制作糖水配料等。
以中医角度来说，据《本草纲目》记载：“白果小苦微甘，性温有小毒，多食令人腹胀”；银杏的果实内含小量氢氰酸毒素，性温，多食令人腹胀，遇热毒性减少，所以生吃或大量进食易引起中毒；多见于小儿；有呕吐、精神萎靡、发热、抽搐等征状。又说“熟食温肺、益气、定喘嗽、缩小便，止白浊，生食降痰，消毒杀虫，嚼浆涂鼻面手足，去鼻疽疱黑干黯皴皱及疥癣疳虫阴虱”。而银杏的种籽，即果仁有暖肺、止喘嗽及减少痰量之功效。特别是对于哮喘、慢性气管及支气管炎及肺结核等病症有明显的疗效。而且对补助泌尿系统有好处，滋阴益肾，可改善尿频。
    """,
    r"""
    杜鹃花属植物俗称杜鹃花，简称杜鹃。当中又名为映山红、满山红、山踯躅、红踯躅、山石榴等的美丽原种，如映山红等美丽种类是中国十大名花之一。全世界的杜鹃花属原种大约有960种，于中国境内有570余种。杜鹃花是尼泊尔的国花，是中国江西省的省花，也是无锡、镇江、三明、长沙、韶关、大理、嘉兴市花，以及台湾台北市、新竹市的市花。
杜鹃花分布非常广泛，北半球大部分地方都有分布，南半球分布于东南亚和北澳大利亚。中国横断山脉周围的云南、四川、西藏以及喜马拉雅山南麓尼泊尔、锡金、印度北阿坎德邦等地分布种类最多的大形杜鹃。其它映山红亚属种类分布较多的地方有中国大陆南部、台湾、印度支那半岛、朝鲜半岛及日本。而南下至东南亚，尤其以新畿内亚岛上蕴藏著约占杜鹃花属三分之一数量的著生杜鹃的族群。北美和欧洲分布较少。南美和非洲分布极少。
70%的杜鹃花种类生长在海拔1700～3700米的地区。
在中国的分布中，云南有245种，西藏有180种，四川有181种（其中贡嘎山地区共有80种，多数为特有种，是中国杜鹃花分布最多的地区之一），广西有60余种，贵州有约60种。
自19世纪中期以来，罗伯特·福琼、约瑟夫·虎克、乔治·福雷斯特、亨利·威尔逊、约瑟夫·洛克等西方人从中国大规模引种杜鹃种源，使得杜鹃花在欧美的分布大大增加，并进而遍布全世界的园林。
    """,
]

questions = [
    "榕树的原产地在哪里",
    "银杏的寿命有多长",
    "银杏的种子叫什么",
    "银杏的种子对于什么疾病有效",
    "杜鹃花分布于哪里",
]

if __name__ == '__main__':
    for question in questions:
        scoreForEachQuestion = []
        resultOfEachQuestion = []
        for text in texts:
            answer = questionAnswering({'question': question, 'context': text})
            scoreForEachQuestion.append(answer['score'])
            resultOfEachQuestion.append(answer['answer'])
        print("Question: %s" % question)
        print("Answer: %s" % resultOfEachQuestion[scoreForEachQuestion.index(max(scoreForEachQuestion))])

    '''
    Question: 榕树的原产地在哪里
    Answer: 中国
    Question: 银杏的寿命有多长
    Answer: 3000年以上
    Question: 银杏的种子叫什么
    Answer: 白果
    Question: 银杏的种子对于什么疾病有效
    Answer: 哮喘
    Question: 杜鹃花分布于哪里
    Answer: 南半球分布于东南亚和北澳大利亚
    '''
