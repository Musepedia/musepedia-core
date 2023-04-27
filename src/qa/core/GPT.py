# -*- coding: utf-8 -*-

from dataclasses import dataclass

import openai

from Config import OPENAI_API_KEY, OPENAI_ORGANIZATION
from src.common.exception.ExceptionHandler import check_length, catch
from src.common.log.GPTLogging import gpt_logging
from src.common.log.ModelLogging import model_logging
from src.common.utils.Template import template
from src.qa.utils.TemplateUtil import TemplateUtil


@dataclass
class Exhibit:
    exhibit_label: str              # 展品名称
    exhibit_descriptions: [str]     # 展品介绍


@dataclass
class GPTContext:
    user_question: str                  # 用户原始输入的问题
    qa_prompt: bool = False             # 是否需要在QA提示符，默认False
    exhibits: [Exhibit] = None          # 展品列表


@dataclass
class GPTCompletion:
    prompt: str             # 根据用户的输入，构造的prompt
    completion: str         # GPT模型接收prompt后生成的内容
    prompt_tokens: int      # prompt占用的token数
    completion_tokens: int  # completion占用的token数


class GPT:
    """
    调用GPT-3.5模型的API，针对用户提问，按模版构造prompt，生成具有一定随机性的回答
    """

    MODEL = 'gpt-3.5-turbo'
    TEMPLATE_DIR_PATH = '../templates/'

    @model_logging('GPT模型')
    def __init__(self, model=MODEL, template_dir_path=TEMPLATE_DIR_PATH):
        self._template_dir_path = template_dir_path
        self._template_util = TemplateUtil(self._template_dir_path)
        self._model = model
        openai.organization = OPENAI_ORGANIZATION
        openai.api_key = OPENAI_API_KEY

    @staticmethod
    def list_model():
        return openai.Model.list()

    @template()
    def create_exhibit_description_prompt(self,
                                          qa_prompt: bool,
                                          exhibit_label: str,
                                          exhibit_descriptions: [str],
                                          variables=None) -> str:
        return self._template_util.render_template('exhibit_introduction.jinja', variables)

    @template()
    def create_exhibit_user_question_prompt(self,
                                            qa_prompt: bool,
                                            exhibit_label_description_dict: dict,
                                            user_question: str,
                                            variables=None) -> str:
        return self._template_util.render_template('exhibit_user_question.jinja', variables)

    @template()
    def create_system_prompt(self,
                             museum_name: str,
                             variables=None) -> str:
        return self._template_util.render_template('system.jinja', variables)

    @catch(Exception)
    def create_user_prompt(self, context: GPTContext) -> str:
        """
        根据用户的输入，选择合适的模板构造prompt

        :param context: 用户的输入，可以支持有上下文的多轮对话
        :return: 根据模版构造的prompt
        """

        if context.user_question != context.exhibits[0].exhibit_label:
            # 当用户问题不是展品名称时，即用户不以展品关键词作为提问时
            exhibit_label_description_dict = {}
            for exhibit in context.exhibits:
                exhibit_label_description_dict[exhibit.exhibit_label] = exhibit.exhibit_descriptions
            return self.create_exhibit_user_question_prompt(context.qa_prompt,
                                                            exhibit_label_description_dict,
                                                            context.user_question)
        else:
            return self.create_exhibit_description_prompt(context.qa_prompt,
                                                          context.exhibits[0].exhibit_label,
                                                          context.exhibits[0].exhibit_descriptions)

    @catch(Exception)
    @check_length(4096)
    @gpt_logging
    def generate(self, user_prompt: str, system_prompt: str) -> GPTCompletion:
        """
        根据构造出的prompt，通过API请求生成回答

        :param user_prompt: 根据用户输入和模板构造出的prompt，作为GPT模型输入的一部分
        :param system_prompt: 系统prompt，作为GPT模型输入的一部分
        :return: GPT模型生成的回答
        """

        if self._model == 'gpt-3.5-turbo':
            completion = openai.ChatCompletion.create(
                model=self._model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )

            return GPTCompletion(user_prompt,
                                 completion.get('choices')[0]['message']['content'],
                                 completion.get('usage')['prompt_tokens'],
                                 completion.get('usage')['completion_tokens'])
        elif self._model == 'text-davinci-003':
            completion = openai.Completion.create(
                model=self._model,
                prompt=user_prompt,
                max_tokens=2048,
                temperature=0.7
            )

            return GPTCompletion(user_prompt,
                                 completion.get('choices')[0]['text'],
                                 completion.get('usage')['prompt_tokens'],
                                 completion.get('usage')['completion_tokens'])
        else:
            # 在后续版本中支持从list_model()方法获取微调过的GPT-3.5模型
            raise Exception('{0}为不支持的模型类型，仅支持[gpt-turbo-3.5|text-davinci-003]'.format(self._model))

    def finetune(self):
        pass


if __name__ == '__main__':
    gpt = GPT()
    user_prompt = gpt.create_user_prompt(GPTContext(user_question='北极熊每天要吃多少东西',
                                                    exhibits=[Exhibit('北极熊',
                                                                      ['北极熊又称 白熊 或 冰熊，是产自北极地区的一种大型肉食性哺乳动物，北极熊是现存体型最大的熊，也是现存陆地上最大的食肉动物，约在四十万年前由古代棕熊演化而来。北极熊是一种能在恶劣酷寒的环境下生存的动物，其活动范围主要在北冰洋、即北极圈附近，而最南则可以在有浮冰出没的地方找到牠们。而最北可以在北纬88度找到牠们，牠们分布在北极点。国际自然保护联盟中的物种存续委员会把北极熊的栖息地划分为十九个地域以做。',
                                                                       '为了加强国际合作和种群管理，在IUCN 北极熊专家组的极力促成之下，1973年所有北极熊的分布国共同签署了《全球北极熊保护协定》，就迁徙种群、北极熊保护和管理做了详细约定1975年，《濒危野生动植物种国际贸易公约》正式生效，北极熊被列入CITES附录Ⅱ，1992年熊科整科被列入CITES附录Ⅱ在所生存的空间里，北极熊位于食物链的最顶层健康的北极熊会拥有极厚的脂肪及毛发，以在北极这种极端严寒的气候中生存其中白色的外表在雪白的雪地上是良好的保护色北极熊是游泳健将，主要在海冰上捕捉海豹为食北极熊是熊科熊属动物科学家已有共识，北极熊是由棕熊演化出来但演化的时间，还有不同的争议之前大部份的认定北极熊是最现代的熊，约15万年前演化出来，但新的DNA技术令有些科学家认为是约60万年前演化出来。',
                                                                       '也有科学家认为，不同区域间的北极熊，基因几乎变化性不够大，因此在面临气候变迁的危机，北极熊缺乏多样性的基因，容易导致他们全面性灭绝北极熊的皮肤其实是黑色，透明的毛发在阳光及冰层的反射下看起来是白色，使牠们能够在冰层上悄悄的跟踪并突袭猎物由于毛发是保暖的关键，当毛发脏污、或因下水而沾有盐份时，会影响其保暖的效能，因此北极熊一般都会注重整理毛发母熊会仔细地舔拭、整理小熊的皮毛，成熊则用全身抖动、雪地上打滚、找淡水水源清洗自己的方式来维持清洁由于小熊的脂肪层不够丰厚、散热体积比率大，在水中容易失温，因此母熊会尽可能不让小熊入海当母熊因为周边海冰不够、食物来源不足、而被迫必须长泳觅食时，跟在母熊后的小熊，往往会溺死在海中由于北极熊在几十万年间已发展为在极端酷寒中，依旧可以有效率保温的体质，牠们的皮毛会有效锁住热气、让体热不致流失，少数的热量、只能从四肢的脚掌及脸部散逸，因此在天气暖化的过程中，也容易造成过热死亡。']),
                                                                ]))
    system_prompt = gpt.create_system_prompt(museum_name='上海自然博物馆')

    print(gpt.generate(user_prompt, system_prompt))
    # GPTCompletion(prompt='结合展品北极熊介绍：北极熊又称 白熊 或 冰熊，是产自北极地区的一种大型肉食性哺乳动物，北极熊是现存体型最大的熊，也是现存陆地上最大的食肉动物，约在四十万年前由古代棕熊演化而来。北极熊是一种能在恶劣酷寒的环境下生存的动物，其活动范围主要在北冰洋、即北极圈附近，而最南则可以在有浮冰出没的地方找到牠们。而最北可以在北纬88度找到牠们，牠们分布在北极点。国际自然保护联盟中的物种存续委员会把北极熊的栖息地划分为十九个地域以做。为了加强国际合作和种群管理，在IUCN 北极熊专家组的极力促成之下，1973年所有北极熊的分布国共同签署了《全球北极熊保护协定》，就迁徙种群、北极熊保护和管理做了详细约定1975年，《濒危野生动植物种国际贸易公约》正式生效，北极熊被列入CITES附录Ⅱ，1992年熊科整科被列入CITES附录Ⅱ在所生存的空间里，北极熊位于食物链的最顶层健康的北极熊会拥有极厚的脂肪及毛发，以在北极这种极端严寒的气候中生存其中白色的外表在雪白的雪地上是良好的保护色北极熊是游泳健将，主要在海冰上捕捉海豹为食北极熊是熊科熊属动物科学家已有共识，北极熊是由棕熊演化出来但演化的时间，还有不同的争议之前大部份的认定北极熊是最现代的熊，约15万年前演化出来，但新的DNA技术令有些科学家认为是约60万年前演化出来。也有科学家认为，不同区域间的北极熊，基因几乎变化性不够大，因此在面临气候变迁的危机，北极熊缺乏多样性的基因，容易导致他们全面性灭绝北极熊的皮肤其实是黑色，透明的毛发在阳光及冰层的反射下看起来是白色，使牠们能够在冰层上悄悄的跟踪并突袭猎物由于毛发是保暖的关键，当毛发脏污、或因下水而沾有盐份时，会影响其保暖的效能，因此北极熊一般都会注重整理毛发母熊会仔细地舔拭、整理小熊的皮毛，成熊则用全身抖动、雪地上打滚、找淡水水源清洗自己的方式来维持清洁由于小熊的脂肪层不够丰厚、散热体积比率大，在水中容易失温，因此母熊会尽可能不让小熊入海当母熊因为周边海冰不够、食物来源不足、而被迫必须长泳觅食时，跟在母熊后的小熊，往往会溺死在海中由于北极熊在几十万年间已发展为在极端酷寒中，依旧可以有效率保温的体质，牠们的皮毛会有效锁住热气、让体热不致流失，少数的热量、只能从四肢的脚掌及脸部散逸，因此在天气暖化的过程中，也容易造成过热死亡。回答游客的提问：北极熊每天要吃多少东西\n', completion='北极熊的饮食量会因其体重、性别和年龄而有所不同。一般来说，北极熊每天需要摄入约2%到5%的体重的食物，也就是成年北极熊每天需要摄入9至18千克的食物。由于北极熊是肉食动物，主要以海豹为食，因此北极熊的饮食量还会受到猎物的可获得性和季节性变化的影响。', prompt_tokens=1387, completion_tokens=160)