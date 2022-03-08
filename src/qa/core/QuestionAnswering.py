# -*- coding: UTF-8 -*-
import torch
import multiprocessing
import numpy as np


from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from common.exception.ExceptionHandler import catch, check_length
from common.exception.TextLengthError import TextLengthError
from common.log.QALogging import qa_logging
from src.qa.core.Answer import Answer

MODEL_PATH = 'src/qa/models/roberta-base-chinese-extractive-qa'


def preload():
    """
    根据模型存储地址，加载tokenizer和模型
    :return: tokenizer与模型
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

    return tokenizer, model


def get_pos_with_logit(start_logits, end_logits):
    """
    计算答案的起始位置和终止位置，以及相应的logits
    :param start_logits: 起始logits
    :param end_logits: 终止logits
    :return: 2个tuple (position, logits)分别代表答案起始位置和终止位置
    """

    startPosWithLogits = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    endPosWithLogits = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

    return startPosWithLogits[0], endPosWithLogits[0]


@catch(TextLengthError, MemoryError, AttributeError)
@check_length(512)
@qa_logging(question_index=2, text_index=3)
def get_possible_answer(tokenizer, model, question, text):
    """
    处理1个问题对应1篇文本，并得到答案
    :param tokenizer: 用于将字符与token映射
    :param model: 模型（现在是Roberta）
    :param question: 问题
    :param text: 待抽取的文本
    :return: 问题对应的答案，如果没有答案，那么返回[CLS]
    """

    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
    outputs = model(**inputs)

    answerStartLogits = outputs.start_logits[0]
    answerEndLogits = outputs.end_logits[0]

    answerParamPair = get_pos_with_logit(answerStartLogits, answerEndLogits)
    answer = Answer(tokenizer, inputs, outputs, answerParamPair)

    return answer


def wrap_get_possible_answer(args):
    """
    封装get_possible_answer()，适用于map()调用
    :param args: 函数get_possible_answer()所需的参数
    :return: tuple 包含答案的分数和文本
    """

    res = get_possible_answer(*args)
    return res.get_score(), res.to_string()


def get_answer(tokenizer, model, question, texts):
    """
    处理1个问题对应多篇文本，将从若干文本中分别抽取答案，同时为每份答案赋予分数（实际是概率），取最高者作为答案
    :param tokenizer: 用于将字符与token映射
    :param model: 模型
    :param question: 问题
    :param texts: 待抽取的文本集合（必须是可迭代对象）
    :return: 问题对应的答案，如果没有答案，那么返回[CLS]
    """

    maxScore = 0
    bestAnswer = ""
    for text in texts:
        possibleAnswer = get_possible_answer(tokenizer, model, question, text)
        if possibleAnswer is None:  # 如果未按预期得到答案，则跳过本轮
            continue
        score = possibleAnswer.get_score()
        if score > maxScore:
            maxScore = score
            bestAnswer = possibleAnswer.to_string()

    return bestAnswer


def get_answer_parallel(tokenizer, model, question, texts, pool_num=4):
    """
    使用并行的方式，处理1个问题对应多篇文本，将从若干文本中分别抽取答案，同时为每份答案赋予分数（实际是概率），取最高者作为答案
    :param tokenizer: 用于将字符与token映射
    :param model: 模型
    :param question: 问题
    :param texts: 待抽取的文本集合（必须是可迭代对象）
    :param pool_num: 并行进程数，默认取4
    :return: 问题对应的答案，如果没有答案，那么返回[CLS]
    """

    with multiprocessing.Pool(pool_num) as pool:
        possibleAnswers = pool.map(wrap_get_possible_answer, [(tokenizer, model, question, texts[i]) for i in range(len(texts))])

    maxScore = 0
    bestAnswer = ""
    for possibleAnswer in possibleAnswers:
        score, answer = possibleAnswer
        if score > maxScore:
            maxScore = score
            bestAnswer = answer

    return bestAnswer


if __name__ == '__main__':
    # question = "银杏的寿命有多长"
    # texts = [
    #     r"""
    #     银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等，其裸露的种子称为白果，叶称蒲扇。属裸子植物银杏门唯一现存物种，和它同门的所有其他物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食物。
    #     """,
    #     r"""
    #     银杏种子可以食用，在中国被称为白果，白果去壳后可煮熟直接食用，制作糖水配料等。以中医角度来说，据《本草纲目》记载：“白果小苦微甘，性温有小毒，多食令人腹胀”；银杏的果实内含小量氢氰酸毒素，性温，多食令人腹胀，遇热毒性减少，所以生吃或大量进食易引起中毒；多见于小儿；有呕吐、精神萎靡、发热、抽搐等征状。又说“熟食温肺、益气、定喘嗽、缩小便，止白浊，生食降痰，消毒杀虫，嚼浆涂鼻面手足，去鼻疽疱黑干黯皴皱及疥癣疳虫阴虱”。而银杏的种籽，即果仁有暖肺、止喘嗽及减少痰量之功效。特别是对于哮喘、慢性气管及支气管炎及肺结核等病症有明显的疗效。而且对补助泌尿系统有好处，滋阴益肾，可改善尿频。
    #     """,
    # ]

    question = '狼和狗有什么关系'
    texts = [
    "狼，或称为灰狼，哺乳纲，犬科，在生物学上与狗为同一物种，为现生犬科动物中体型最大的物种狼这个物种曾是地球上分布地区最广的哺乳动物，包括北美和欧亚大陆，但如今在西欧、墨西哥与美国大部份地区已然绝迹它们主要栖息在荒野或偏远地区，但并不限于此由于人类厌恶狼对豢养牲畜不顾一切的猎捕行为、以及害怕被狼攻击的恐惧、栖息地大量的破坏，其栖息地已经缩减了三分之一目前，狼主要分布于亚洲、欧洲、北美和中东它们是生态系统原有的一部分，各地不同生态系统的多样性，反映了狼这个物种的适应能力这其中包括而不限于森林、沙漠、山地、寒带草原、西伯利亚针叶林、草地虽然就整个物种而言，狼被世界自然保护联盟列为绝种威胁程度最小的等级，然而在某些地区，不同亚种的狼被列为濒临绝种或是受绝种威胁的动物现今在很多地区，狼仍然因为活动或是被视为对牲口威胁的原因，而遭受捕猎狼是社会性的猎食动物，狼群以核心家庭的形式组成，包括一对配偶、及其子女，有时也包括收养的未成年幼狼狼属于典型的食物链上层掠食者。",
    "它们通常群体行动，由于狼会捕食羊等家畜，因此直到20世纪末期前都被人类大量捕杀，一些亚种如日本狼、纽芬兰狼等都已经绝种，虽然有一些其它亚种已经确认，但亚种的确切数量仍旧未定灰狼是体型瘦长，有力的动物有大而深的肋骨架、倾斜的背、腹部内缩、颈部肌肉有力四肢长而强健，和有点小的脚掌每只前掌各有五趾，后掌有四趾前肢看起来像是压入胸腔，肘部向内，而脚掌向外雌狼的前额与嘴比较窄，颈部较薄、腿较短、肩部不如雄狼厚实就这样的体型而言，狼非常强壮，可以将冰冻的麋鹿翻转狼的脚掌可以轻易适应各种类型的地面，特别是雪地它们的足趾之间有一点蹼，使它们在雪地上行动能比猎物更为方便狼是趾行性动物，体重能很好地分布在积雪上它们的前脚掌比后脚掌略大，掌上有五个趾，后脚掌没有上趾掌上的毛和略钝的爪帮助它们抓住湿滑的地面特殊的血管保护狼的脚掌不会在雪地中冻伤，在趾间的腺体分泌会在脚印上留下气味，帮助狼记录自己的行踪，同时也提供线索让其它的狼知道自己的所在与犬不同，狼在脚掌的肉垫上没有汗腺。",
    "在寒冷的天气中狼可以减少血流接近皮肤，以保存体温脚掌垫保暖的调节独立于与身体其馀部份，当掌垫接触冰雪时，可以维持在略高于组织冻伤的温度成年狼的肠道长460-575公分，对身长的比例为4.13-4.62，胃能容纳7-9公斤的食物和多达7.5公升的水肝相当大，雄狼的肝重0.7-1.9公斤，雌狼为 0.68-0.82公斤狼头相当大而重，前额很宽，吻突长呈钝状，下颚强而有力耳朵相对小呈三角形狼的头通常于与背脊同高，只有示警时才会将头抬高牙齿重而大，虽然不如非洲鬣狗般奇特，但比现存犬科其它物种的牙，更适合咬碎骨头狼的咬合压力高达1500 lbf/in2，德国牧羊犬仅有 750 lbf/in2这种力道足以咬碎多数的骨头它们的体形类似德国牧羊犬或哈士奇，但从orbital angle  40°–45° 而非 53°–60° 可以分辨同时狼的头与牙齿都比较大与郊狼相比，狼比较大，嘴也比较宽耳朵较短，和比例上较小的脑、脚掌垫上缺乏汗腺与亚洲胡狼相比，狼体型较大也较重，腿比例也较长，躯干比较短、尾巴较长。",
    "除了某些大型犬之外，狼是现行犬科动物中体型最大的物种，其体重和大小依据它在全球分布地区的不同，有很大差异如Bergmann Rule 所预测的，随著分布的纬度愈高，狼的体型体重也愈大通常而言，狼体长105-160公分，肩高80-85公分，尾的长度约为头与身体的2/3，29-50公分狼体重随地域分布有区别，平均来说，北美狼为75公斤，欧亚狼为50公斤，印度、阿拉伯狼为25公斤，北非狼仅有13公斤在狼群中，母狼比公狼约轻2.2-4.5公斤，体重超过54公斤的狼很罕见，虽然特别大的狼只个体在前苏联、阿拉斯加和加拿大都发现过，北美洲的最高纪录是1939年7月12日在阿拉斯加猎捕到的狼，体重达90公斤而欧亚大陆有案可查的最重的野生狼，是二次世界大战后在乌克兰波尔塔瓦州所猎杀的，体重高达86公斤最小的狼是阿拉伯狼，母狼成熟期也只有10公斤重狼厚重的毛有两层，外层长、粗而硬，主要用于抵御水与灰尘内层则致密与防水内层毛在每年的春末夏初时会脱落，狼会摩擦岩石或树木来促进这层毛的脱落，内层毛在秋天又长回来。",
    "母狼冬季的毛都在春天的时候比公狼换得晩，北美洲的狼毛通常比欧洲的要柔软而长狼的冬毛有极佳的御寒效果，狼在北方的气候中，可以在−40°气温下，于空旷的地方休息，它们会将头放在后腿间，并用尾部盖住脸狼毛隔热的效果比狗毛好，如同狼獾它的皮毛不会因为呼吸温暖的空气而集结冰温暖区域的狼，其狼毛比北方狼粗而少一般来说，雌狼肢体上的毛比较平顺，并随年纪增加更为明显年纪较大的狼在尾部末端、鼻子和前额会有较多灰毛背部中央的毛长约60-70公厘，肩部外层的毛通常不超过90公厘，但可长达110-130公厘狼毛的颜色有很大的变化，从灰色到灰褐色，白褐色和黑色这些颜色通常混杂在一起，当然单一颜色的狼或狼群也并非稀有，通常是白色或黑色狼的嗅觉不如某些猎犬，在上风处2-3公里远能闻到腐肉虽然它们会跟踪新的足迹，但很少能捉到躲藏的野兔或鸟类已知捕捉到的狼可以用嗅觉知道餵养它们的人刚吃了甚么食物它们的听觉非常灵敏，可以听到频率高达 26kHz 的声音，比狐狸好。",
    "虽然它们夜视能力是犬科动物中最好的，但它们的视力不如猫类灰狼的亚种分类非常有争议一般相信过去曾经有约50个亚种存在然而，一个新而被广泛接受的列表将狼分为16种现存的亚种和2种最近灭绝的亚种 这种分类法综合考虑了解剖学上、分布上和不同狼群迁徙习性上的特点 西班牙狼也有可能是个独立亚种 近几年的基因研究发现印度狼与喜马拉雅狼是独立的物种在较早的的文献中，狼群通常被描述为具有严格的社会阶级结构，领头的狼对偶是经由争夺爬到社会阶梯结构的顶层其次是服从阶层、以及最低阶层这样的说法很大程度来自观察捕获的、彼此没有血亲关联的狼，它们彼此会为了地位而打斗和竞争同时捕获的狼由于没有疏散躲避的空间，彼此争斗也比在自然环境中更频繁在野外的自然环境中，狼群比核心家庭更紧密些，其基本社会单元是狼的一对配偶，以及其子女北方狼群的群体倾向于比非洲野犬和斑鬣狗更大、也较复杂，也比郊狼的群体稳定南方狼群的社会行为比较倾向单独或成对生活，与郊狼和澳洲野犬比较接近。",
    "狼群鲜少接受其它狼加入，它们通常会杀死陌生的狼，会被狼群接受的狼都在1-3岁间，被杀死的都是成年狼被狼群接受是个冗长的过程，包括为期长达数周的试探、被非致命性的攻击、以了解新加入者是否可以信任在猎物丰富、迁移、产子等时候，狼群可能会暂时联合纪录中，从五个月到五岁之间的狼都可能离开原生狼群，以发展自己的家庭，但平均年纪是从11-24个月促成离开狼群的机制包含性发育成熟、以及在狼群内部食物的竞争和繁殖狼是具有高度的领域性的动物，通常他们建立的领域超过它们实际生存所需，以确保持续的猎物供应领域的大小依据可捕获猎物的数量决定，猎物丰富时，栖息狼群的领域就比较小，狼群会不断地移动搜寻猎物，每天约涵盖其领域的9% ，他们大约花50%的时间在其核心领域，约35平方公里在其领域周围，猎物密度倾向于远高于其领域中即便领域周遭的猎物较多，除非猎物非常稀缺，狼倾向于避免在领域边界狩猎，因为可以避免致命地遭遇相邻领域的狼群当狼群中的幼狼长到6个月大时，由于需要的营养与成年狼相当，其领域会扩增。",
    "最大的纪录在阿拉斯加州的一个狼群占有6272平方公里在某些区域狼会适应其猎物的迁移季节，而改变其领域狼以气味记号、直接攻击和狼嚎来划分和防卫其领域狼用尿液、排便、摩擦地面以留下气味记号，来标明领地气味记号间隔约为240公尺遍布整个领域，这样的标记约可维持2-3周，记号通常做在岩石、树木、或大型动物的骨头上如果气味记号与狼嚎没有阻止陌生狼群进入其领域，就会发生激烈的遭遇打斗领域争夺是狼致死的主要原因之一，在明尼苏达州和迪纳利国家公园和保留区所做的一项研究显示 14-65% 狼死亡原因来自其它狼的狩猎行为，事实上91%的狼死亡发生在与其它狼群领域边界3.2公里的范围内由于侵入的结果可能是致命的，这种侵入被认为是绝望的和蓄意的攻击行为狼居住密度低的区域，狼通常倾向单一配偶成偶的狼只要配偶还在，绝大多数会终生相伴如果狼的配偶死亡，它会很快重建新配偶关系由于雄狼数量通常多于雌狼，所以没有配对的雌狼很罕见多于一个配偶的关系也可能发生，但这主要发生在狼只被捕豢养的状态。",
    "已知豢养的狼在 9-10 个月就可以繁殖，而野生狼最早的繁殖纪录也要两岁雌狼每年都可以繁殖，通常每年一胎与郊狼不同的是，狼繁殖力不会衰老直到死亡近亲交配鲜少发生，虽然在加拿大萨克其万省和 Isle Royale 有这样的纪录狼通常在暮冬发情，较年长、有子女的雌狼发情时间比年轻雌狼早约 2-3 周发情前，狼群可能暂时解散直到交配期结束愿意接受配偶时，雌狼会将尾巴偏向一边，露出生殖器官交配时狼会有交配姿势，时间 5-36 分钟不等由于狼的发情期为时仅一个月，与狗不同的是，雄狼不会放弃配偶去找其它雌狼雌狼在怀孕期间会留在狼穴，远离其领域的边界区域，以减少与其它狼群冲突的机会较年长的雌狼通常在前一胎的狼穴生产，年轻雌狼则通常选择靠近其出生地的附近怀孕为期 62-75 天，幼狼通常在夏季出生，每胎平均 5-6 只幼狼，14-17 只发生率为 1%每胎的数量倾向随猎物的丰富增加相对犬科其它物种，每胎数量少时，幼狼体型较大与母狼不同，公狼不会反刍食物给幼狼，它会从猎杀中带回食物。"
    ]

    tokenizer, model = preload()
    print(get_answer_parallel(tokenizer, model, question, texts))
    print(get_answer(tokenizer, model, question, texts))
