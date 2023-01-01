# -*- coding: UTF-8 -*-
import torch
import rocketqa

from transformers import DPRReaderTokenizer, DPRReader

from common.exception.ExceptionHandler import catch

MODEL_PATH = 'src/qa/models/dpr-reader-multiset-base'


@catch(Exception)
def preload():
    """
    根据模型存储地址，加载tokenizer和模型
    :return: tokenizer与模型
    """

    tokenizer = DPRReaderTokenizer.from_pretrained(MODEL_PATH)
    model = DPRReader.from_pretrained(MODEL_PATH)

    return tokenizer, model


@catch(Exception)
def get_top_k_text(tokenizer: DPRReaderTokenizer, model: DPRReader, question: str, texts: [str], titles: [str], k: int) -> [str]:
    """
    从texts中找出最可能包含question对应回答的k个文本
    :param tokenizer: 向量化问题与文本
    :param model: 模型，这里采用DPR模型
    :param question: 问题
    :param texts: 可能包含回答的文本集合
    :param titles: 每个文本的标题
    :param k: top k文本
    :return: k个最可能包含回答的文本集合
    """

    inputs = tokenizer(questions=[question] * len(texts),
                       titles=titles,
                       texts=texts,
                       return_tensors='pt',
                       padding=True,
                       truncation=True)
    outputs = model(**inputs)
    logits = outputs.relevance_logits

    _, indices = torch.sort(logits, descending=True)

    return [texts[indices[i]] for i in range(k)]


@catch(Exception)
def temp(model, question, texts, titles):
    assert len(texts) == len(titles)
    res = model.matching(query=[question] * len(texts), para=texts, title=titles)
    res = list(res)

    return texts[res.index(max(res))]


if __name__ == '__main__':
    question = "When did ginkgo first appear in China"
    texts = ["Ginkgos are large trees, normally reaching a height of 20–35 m (66–115 ft),[15] with some specimens in China being over 50 m (165 ft). The tree has an angular crown and long, somewhat erratic branches, and is usually deep-rooted and resistant to wind and snow damage. Young trees are often tall and slender, and sparsely branched; the crown becomes broader as the tree ages. A combination of resistance to disease, insect-resistant wood, and the ability to form aerial roots and sprouts makes ginkgos durable, with some specimens claimed to be more than 2,500 years old.[16]",
             "The leaves are unique among seed plants, being fan-shaped with veins radiating out into the leaf blade, sometimes bifurcating (splitting), but never anastomosing to form a network.[17] Two veins enter the leaf blade at the base and fork repeatedly in two; this is known as dichotomous venation. The leaves are usually 5–10 cm (2–4 in), but sometimes up to 15 cm (6 in) long. The old common name, maidenhair tree, derives from the leaves resembling pinnae of the maidenhair fern, Adiantum capillus-veneris.[citation needed] Ginkgos are prized for their autumn foliage, which is a deep saffron yellow. Leaves of long shoots are usually notched or lobed, but only from the outer surface, between the veins. They are borne both on the more rapidly growing branch tips, where they are alternate and spaced out, and also on the short, stubby spur shoots, where they are clustered at the tips. Leaves are green both on the top and bottom[18] and have stomata on both sides.[19] During autumn, the leaves turn a bright yellow and then fall, sometimes within a short space of time (one to 15 days).[20]",
             "Ginkgo biloba, commonly known as ginkgo or gingko (/ˈɡɪŋkoʊ, ˈɡɪŋkɡoʊ/ GINK-oh, -⁠goh)[5][6] also known as the maidenhair tree,[7] is a species of tree native to China. It is the last living species in the order Ginkgoales, which first appeared over 290 million years ago. Fossils very similar to the living species, belonging to the genus Ginkgo, extend back to the Middle Jurassic approximately 170 million years ago.[2] The tree was cultivated early in human history and remains commonly planted. Ginkgo leaf extract is commonly used as a dietary supplement, but there is no scientific evidence that it supports human health or is effective against any disease.[8][9]"]
    titles = ["description", "leaves", "introduction of ginkgo"]

    question_chinese = "银杏寿命有多长"
    texts_chinese = ["银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等[4]，其裸露的种子称为白果，叶称蒲扇[5]。属裸子植物银杏门惟一现存物种，和它同门的其他所有物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，常作为道路景观树种，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食材。",
                     "和它相亲的银杏类植物在两亿七千万年前的二叠纪时就已经生成，属于银杏门。晚三叠纪时，银杏类植物快速发展，之后的侏罗纪和早白垩纪达到了鼎盛时期，银杏类的五个科同时存在，除赤道外广泛分布于世界各大洲。但白垩纪后期被子植物迅速崛起时，银杏类像其它裸子植物一样也急剧衰落。晚白垩纪后除个别发现外，银杏科以外的银杏类植物已基本绝迹。晚白垩纪和古近纪，银杏（主要为银杏属Ginkgo 和似银杏属Ginkgoites）在欧亚大陆和北美高纬度地区呈环北极分布，渐新世时由于寒冷气候不断向南迁徙，并在此之后不断衰落。银杏在中新世末在北美消失，上新世晚期在欧洲消失。250多万年前发生第四纪冰河时期，令银杏数量继续减少，面临绝灭的危机，而中国南部因地理位置适合和气候温和，成为银杏的最后栖息地。[6]中国的银杏大化石纪录始于始新世；日本直至上新世，甚至更新世早期都有银杏叶化石发现，但没有发现繁殖器官。而现在的银杏是这个门的植物中生存至今的唯一成员，因此又被称为“活化石（孑遗植物）”。[7]",
                     "银杏在中国古代称为“银果”，如今又称为“白果”。“白果”这个名称直接借入越南语，依汉越音发音为“bạch quả”；“银杏”这个名称则借入朝鲜语和日语，根据朝鲜语的汉字音分别读作“은행”（eunhaeng）和“ぎんなん”（ginnan）。学名的“Ginkgo”来源于日本民间，日语中的汉字常有多种读音，而“銀杏”也可发音为“ぎんきょう”（ginkyō）。1690年，德国植物学家恩格尔贝特·肯普弗（Engelbert Kaempfer）成为第一个发现银杏的西方人，在其著作《异域采风记》（Amoenitates Exoticae，1712年）中记录了银杏的发音。 一般认为，他写的“y”被误读成了“g”，而这个误读后来被瑞典生物学家林奈继承，并一直沿用至今[12]。但也有学者认为，在德语中读作“y”的音通常写作“j”，而坎普法来自德国北部的莱姆戈，在当地方言中会用“g”代替“j”。还有学者认为，坎普法的一位助手（今村源右衛門英生，Genemon Imamura Eisei，也作 Ichibei）来自长崎，而“kgo”一词刚好准确地展示了当时（17世纪末）长崎地区的日语方言读音。"]
    titles_chinese = ["介绍", "演化", "名称"]

    repeat = 1

    # tokenizer, model = preload()
    # print(get_top_k_text(tokenizer, model, question_chinese, texts_chinese * repeat, titles_chinese * repeat, 2))

    model = rocketqa.load_model('/Users/sornk/Downloads/zh_dureader_de/config.json')
    # res = model.matching(query=model.encode_query(question), para=model.encode_para(texts), title=titles)

    texts_chinese[0] = texts_chinese[0] * 2

    print(temp(model, question_chinese, texts_chinese, titles_chinese))
