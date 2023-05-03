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
                                                    exhibits=[Exhibit('北极熊', [
                                                        '为了加强国际合作和种群管理，在IUCN 北极熊专家组的极力促成之下，1973年所有北极熊的分布国共同签署了《全球北极熊保护协定》，就迁徙种群、北极熊保护和管理做了详细约定1975年，《濒危野生动植物种国际贸易公约》正式生效，北极熊被列入CITES附录Ⅱ，1992年熊科整科被列入CITES附录Ⅱ在所生存的空间里，北极熊位于食物链的最顶层健康的北极熊会拥有极厚的脂肪及毛发，以在北极这种极端严寒的气候中生存其中白色的外表在雪白的雪地上是良好的保护色北极熊是游泳健将，主要在海冰上捕捉海豹为食北极熊是熊科熊属动物科学家已有共识，北极熊是由棕熊演化出来但演化的时间，还有不同的争议之前大部份的认定北极熊是最现代的熊，约15万年前演化出来，但新的DNA技术令有些科学家认为是约60万年前演化出来。',
                                                        '北极熊又称 白熊 或 冰熊，是产自北极地区的一种大型肉食性哺乳动物，北极熊是现存体型最大的熊，也是现存陆地上最大的食肉动物，约在四十万年前由古代棕熊演化而来北极熊是一种能在恶劣酷寒的环境下生存的动物，其活动范围主要在北冰洋、即北极圈附近，而最南则可以在有浮冰出没的地方找到牠们而最北可以在北纬88度找到牠们，牠们分布在北极点国际自然保护联盟中的物种存续委员会把北极熊的栖息地划分为十九个地域以做科学研究目的，分布在五个国家：阿拉斯加、加拿大、俄罗斯、挪威、格陵兰由于浮冰是北极熊捕食海豹和鱼类 赖以生存的重要栖地，因此这些地点皆是海冰布集的区域、或是母北极熊生育小北极熊的的育哺洞穴）的重要所在地牠们的主要活动中心为：北极熊一般被定义为独居性动物），除了养育小熊的母熊、或交配期间公母熊在一起的两三个星期间外，他们几乎都独自活动、到处捕猎，因此范围可以非常之大根据科学家追踪个别母熊的记录，发现不同母熊的活动范围有明显不同，小到200平方公里、大可到96万平方公里。',
                                                        '北极熊在猎捕到海豹后，会先食用其身上的脂肪层，因为那是摄取惟生所需的热量最重要来源擅于捕猎的北极熊、尤其是壮硕的公熊，有时会把猎物的脂肪层吃完后，其他弃而不食，就会成为其他年轻或老年公熊、母熊及小熊们在饥饿时的食物来源北极熊的食馀，尤其在严酷的北极冬天时、是其他极带动物赖以生存的食粮来源，例如北极狐因此科学家会观察到，成年的大北极熊身后，有时会跟著一只雪白的北极狐北极熊主要的食物来源有以下：环斑海豹是海豹中体型最小的海豹，成年长约4英尺、重约150磅母熊、尤其是养有小熊的母熊，大多是猎捕环斑海豹为生母熊带著小熊刚离哺育洞穴时，也刚好是环斑海豹在海冰上产小海豹的时间常见有影片、北极熊不停地在冰上嗅寻后，忽然人立起来，快速以两只前掌用力击破地上的冰雪后，整个上半身立刻钻进洞去追捕猎物，只露出㘣㘣屁股和两只后脚撑在洞口，这时的猎物，多是环斑海豹小海豹刚出生时的脂肪层不足，北极熊的最终目的是希望能猎取到成年母海豹，惟成年海豹的警觉性高、不易捕捉世界上现约有250万只环斑海豹。',
                                                        '几十万年来，在北极酷寒的环境中，北极熊已演化全几乎完全的肉食性动物，少数食用的植物为海草、或夏天冻原间偶有的蓝苺等，但只能用以补充其矿物质，无法提供生存所需的热量他们赖以生存的热量，几乎百分之九十需要仰赖猎捕不同的海豹北极熊的嗅觉非常敏锐，是猎犬的七倍左右，可闻到几公里以外的海豹，并可找到一两公尺以下雪层的海豹北极熊是很聪明、充满好奇、学习力强的动物牠们的猎捕技巧完全学习自母熊，有冰上追踪、水面追踪、游泳跟踪、耐性守候、破冰袭击、直接攻击、声东击西等等不一而足由于海冰可以瞬息万变、海豹的警觉性很高，他们必须要在冰上世界学习去灵活地运用各种不同的技巧捕猎，才能生存下去一般人可能认为北极熊捕食海豹轻而易举，事实上，在稳固又适宜的海冰、或冰间湖上，北极熊猎捕五六次的成功机率，可能只有一次而在开放海域上的海冰，猎捕成功的机率会降到佰分之五以下每二十次的猎捕，可能只有一次会成功在海中海豹远较北极熊灵活、下潜速度快，此外北极熊无法闭气过久，因此北极熊几乎无法在海中追捕到海豹。',
                                                        '尽管身躯庞大，为了捕捉海豹，北极熊其实相当灵活，奔跑的时速可以达到25英哩由于需要在不同的海冰层、岛屿、或冰间湖间捕捉海豹，北极熊也擅长游泳，能以6英哩的速度、一天游约60英哩远他们游泳时，前面两掌如同船桨般、在水中交互滚动前进由于全球暖化，各海域的海冰在不断地消褪中、北极熊有时为了找寻海冰上的海豹，一游需要好几天无以进食，消耗极大的体力科学家估计，全世界的北极熊现今只剩不到2万5仟只，面临绝种的危机北极熊是熊类中，生产量最低的种类之一由于小熊赖母熊以维生的时间很长，母熊一生中，大约只会生产5胎，而只数平均约两只，远低于其先祖灰熊北极熊养育后代的责任，完全落在母熊身上，一般可长达2年半、甚至有3年的记录成年的公熊在春天时会积极地找可交配的母熊，有时追随一只母熊的气味，可长达数日甚至上周母熊一般不会主动，甚至会测试公熊的体力来决定是否要与之交配。'
                                                    ])]))
    system_prompt = gpt.create_system_prompt(museum_name='上海自然博物馆')

    print(gpt.generate(user_prompt, system_prompt))
    # GPTCompletion(prompt='结合展品北极熊介绍：北极熊又称 白熊 或 冰熊，是产自北极地区的一种大型肉食性哺乳动物，北极熊是现存体型最大的熊，也是现存陆地上最大的食肉动物，约在四十万年前由古代棕熊演化而来。北极熊是一种能在恶劣酷寒的环境下生存的动物，其活动范围主要在北冰洋、即北极圈附近，而最南则可以在有浮冰出没的地方找到牠们。而最北可以在北纬88度找到牠们，牠们分布在北极点。国际自然保护联盟中的物种存续委员会把北极熊的栖息地划分为十九个地域以做。为了加强国际合作和种群管理，在IUCN 北极熊专家组的极力促成之下，1973年所有北极熊的分布国共同签署了《全球北极熊保护协定》，就迁徙种群、北极熊保护和管理做了详细约定1975年，《濒危野生动植物种国际贸易公约》正式生效，北极熊被列入CITES附录Ⅱ，1992年熊科整科被列入CITES附录Ⅱ在所生存的空间里，北极熊位于食物链的最顶层健康的北极熊会拥有极厚的脂肪及毛发，以在北极这种极端严寒的气候中生存其中白色的外表在雪白的雪地上是良好的保护色北极熊是游泳健将，主要在海冰上捕捉海豹为食北极熊是熊科熊属动物科学家已有共识，北极熊是由棕熊演化出来但演化的时间，还有不同的争议之前大部份的认定北极熊是最现代的熊，约15万年前演化出来，但新的DNA技术令有些科学家认为是约60万年前演化出来。也有科学家认为，不同区域间的北极熊，基因几乎变化性不够大，因此在面临气候变迁的危机，北极熊缺乏多样性的基因，容易导致他们全面性灭绝北极熊的皮肤其实是黑色，透明的毛发在阳光及冰层的反射下看起来是白色，使牠们能够在冰层上悄悄的跟踪并突袭猎物由于毛发是保暖的关键，当毛发脏污、或因下水而沾有盐份时，会影响其保暖的效能，因此北极熊一般都会注重整理毛发母熊会仔细地舔拭、整理小熊的皮毛，成熊则用全身抖动、雪地上打滚、找淡水水源清洗自己的方式来维持清洁由于小熊的脂肪层不够丰厚、散热体积比率大，在水中容易失温，因此母熊会尽可能不让小熊入海当母熊因为周边海冰不够、食物来源不足、而被迫必须长泳觅食时，跟在母熊后的小熊，往往会溺死在海中由于北极熊在几十万年间已发展为在极端酷寒中，依旧可以有效率保温的体质，牠们的皮毛会有效锁住热气、让体热不致流失，少数的热量、只能从四肢的脚掌及脸部散逸，因此在天气暖化的过程中，也容易造成过热死亡。回答游客的提问：北极熊每天要吃多少东西\n', completion='北极熊的饮食量会因其体重、性别和年龄而有所不同。一般来说，北极熊每天需要摄入约2%到5%的体重的食物，也就是成年北极熊每天需要摄入9至18千克的食物。由于北极熊是肉食动物，主要以海豹为食，因此北极熊的饮食量还会受到猎物的可获得性和季节性变化的影响。', prompt_tokens=1387, completion_tokens=160)