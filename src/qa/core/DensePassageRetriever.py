# -*- coding: UTF-8 -*-
import torch
import rocketqa

from transformers import DPRReaderTokenizer, DPRReader

from Config import USE_GPU
from src.common.exception.ExceptionHandler import catch
from src.common.log.ModelLogging import model_logging


class DensePassageRetriever:
    """
    基于稠密段落检索的方法，作为开放域问答检索器的一种实现方案，相比ElasticSearch作为检索器，其精度较高但效率较低
    """

    FACEBOOK_DPR_MODEL_PATH = '../models/dpr-reader-multiset-base'
    ROCKETQA_DPR_MODEL_PATH = '../models/zh_dureader_de/config.json'

    def __init__(self, model_type='rocket', model_path=ROCKETQA_DPR_MODEL_PATH):
        self._model_type = model_type.lower()
        self._use_cuda = torch.cuda.is_available() and USE_GPU
        self._device = 'cuda:0' if self._use_cuda else 'cpu'
        if self._model_type == 'facebook':
            self._model_path = self.FACEBOOK_DPR_MODEL_PATH
        else:
            self._model_path = model_path
        self._tokenizer, self._model = self.preload(self._model_type)

    @model_logging('DPR模型')
    def preload(self, model_type: str):
        """
        根据模型存储地址，加载tokenizer和模型

        :param model_type: 模型类型，支持Facebook的DPR模型和Paddle的RocketQA模型，对于中文语料，默认使用RocketQA模型
        :return: tokenizer与模型构成的元组
        """

        tokenizer = None
        model = None

        if model_type == 'facebook':
            tokenizer = DPRReaderTokenizer.from_pretrained(self._model_path)
            model = DPRReader.from_pretrained(self._model_path).to(self._device)
        elif model_type == 'rocket':
            if self._use_cuda:
                model = rocketqa.load_model(self._model_path, use_cuda=True, device_id=0)
            else:
                model = rocketqa.load_model(self._model_path)
        else:
            raise Exception('{0}为不支持的模型类型，仅支持[facebook|rocket]'.format(model_type))

        return tokenizer, model

    @catch(Exception)
    def get_top_k_text(self, question: str, texts: [str], titles: [str], k: int) -> [str]:
        """
        从texts中找出最可能包含question对应回答的k个文本

        :param question: 问题
        :param texts: 可能包含回答的文本集合
        :param titles: 每个文本的标题
        :param k: top k文本
        :return: k个最可能包含回答的文本集合
        """

        top_k_texts = []
        if self._model_type == 'facebook':
            inputs = self._tokenizer(questions=[question] * len(texts),
                               titles=titles,
                               texts=texts,
                               return_tensors='pt',
                               padding=True,
                               truncation=True).to(self._device)
            outputs = self._model(**inputs)
            logits = outputs.relevance_logits

            _, indices = torch.sort(logits, descending=True)
            top_k_texts = [texts[indices[i]] for i in range(k)]
        elif self._model_type == 'rocket':
            top_k_texts = list(self._model.matching(query=[question] * len(texts), para=texts, title=titles))
            top_k_texts = texts[top_k_texts.index(max(top_k_texts))]
        else:
            raise Exception('{0}为不支持的模型类型，仅支持[facebook|rocket]'.format(self._model_type))

        return top_k_texts


if __name__ == '__main__':
    question_chinese = "银杏寿命有多长"
    texts_chinese = ["银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等[4]，其裸露的种子称为白果，叶称蒲扇[5]。属裸子植物银杏门惟一现存物种，和它同门的其他所有物种都已灭绝，因此被称为植物界的“活化石”。已发现的化石可以追溯到2.7亿年前。银杏原产于中国，现广泛种植于全世界，常作为道路景观树种，并被早期引入人类历史。它有多种用途，可作为传统医学用途和食材。",
                     "和它相亲的银杏类植物在两亿七千万年前的二叠纪时就已经生成，属于银杏门。晚三叠纪时，银杏类植物快速发展，之后的侏罗纪和早白垩纪达到了鼎盛时期，银杏类的五个科同时存在，除赤道外广泛分布于世界各大洲。但白垩纪后期被子植物迅速崛起时，银杏类像其它裸子植物一样也急剧衰落。晚白垩纪后除个别发现外，银杏科以外的银杏类植物已基本绝迹。晚白垩纪和古近纪，银杏（主要为银杏属Ginkgo 和似银杏属Ginkgoites）在欧亚大陆和北美高纬度地区呈环北极分布，渐新世时由于寒冷气候不断向南迁徙，并在此之后不断衰落。银杏在中新世末在北美消失，上新世晚期在欧洲消失。250多万年前发生第四纪冰河时期，令银杏数量继续减少，面临绝灭的危机，而中国南部因地理位置适合和气候温和，成为银杏的最后栖息地。[6]中国的银杏大化石纪录始于始新世；日本直至上新世，甚至更新世早期都有银杏叶化石发现，但没有发现繁殖器官。而现在的银杏是这个门的植物中生存至今的唯一成员，因此又被称为“活化石（孑遗植物）”。[7]",
                     "银杏在中国古代称为“银果”，如今又称为“白果”。“白果”这个名称直接借入越南语，依汉越音发音为“bạch quả”；“银杏”这个名称则借入朝鲜语和日语，根据朝鲜语的汉字音分别读作“은행”（eunhaeng）和“ぎんなん”（ginnan）。学名的“Ginkgo”来源于日本民间，日语中的汉字常有多种读音，而“銀杏”也可发音为“ぎんきょう”（ginkyō）。1690年，德国植物学家恩格尔贝特·肯普弗（Engelbert Kaempfer）成为第一个发现银杏的西方人，在其著作《异域采风记》（Amoenitates Exoticae，1712年）中记录了银杏的发音。 一般认为，他写的“y”被误读成了“g”，而这个误读后来被瑞典生物学家林奈继承，并一直沿用至今[12]。但也有学者认为，在德语中读作“y”的音通常写作“j”，而坎普法来自德国北部的莱姆戈，在当地方言中会用“g”代替“j”。还有学者认为，坎普法的一位助手（今村源右衛門英生，Genemon Imamura Eisei，也作 Ichibei）来自长崎，而“kgo”一词刚好准确地展示了当时（17世纪末）长崎地区的日语方言读音。"]
    titles_chinese = ["介绍", "演化", "名称"]

    repeat = 1

    dense_passage_retriever = DensePassageRetriever(model_type='facebook')
    print(dense_passage_retriever.get_top_k_text(question_chinese, texts_chinese, titles_chinese, 2))
