from dataclasses import dataclass

import openai

from Config import OPENAI_API_KEY, OPENAI_ORGANIZATION
from src.common.exception.ExceptionHandler import check_length, catch
from src.common.log.GPTLogging import gpt_logging
from src.common.log.ModelLogging import model_logging
from src.common.utils.Template import template
from src.qa.utils.TemplateUtil import TemplateUtil


@dataclass
class GPTContext:
    user_question: str                  # 用户原始输入的问题
    qa_prompt: bool = False             # 是否需要在QA提示符，默认False
    exhibit_label: str = None           # 展品名称，默认为空
    exhibit_description: str = None     # 展品介绍，默认为空


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
                                          exhibit_description: str,
                                          variables=None) -> str:
        return self._template_util.render_template('exhibit_introduction.jinja', variables)

    @template()
    def create_exhibit_user_question_prompt(self,
                                            qa_prompt: bool,
                                            exhibit_description: str,
                                            user_question: str,
                                            variables=None) -> str:
        return self._template_util.render_template('exhibit_user_question', variables)

    @template()
    def create_system_prompt(self,
                             museum_name: str,
                             museum_description: str,
                             variables=None) -> str:
        return self._template_util.render_template('system.jinja', variables)

    @catch(Exception)
    def create_user_prompt(self, context: GPTContext) -> str:
        """
        根据用户的输入，选择合适的模板构造prompt

        :param context: 用户的输入，可以支持有上下文的多轮对话
        :return: 根据模版构造的prompt
        """

        if context.user_question != context.exhibit_label:
            # 当用户问题不是展品名称时，即用户不以展品关键词作为提问时
            return self.create_exhibit_user_question_prompt(context.qa_prompt,
                                                            context.exhibit_description,
                                                            context.user_question)
        else:
            return self.create_exhibit_description_prompt(context.qa_prompt,
                                                          context.exhibit_label,
                                                          context.exhibit_description)

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
    user_prompt = gpt.create_user_prompt(GPTContext(user_question='银杏',
                                                    exhibit_label='银杏',
                                                    exhibit_description='银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等[4]，其裸露的种子称为白果，叶称蒲扇[5]。'))
    system_prompt = gpt.create_system_prompt(museum_name='上海自然博物馆',
                                             museum_description='上海自然博物馆以“自然·人·和谐”为主题，通过“演化的乐章”、“生命的画卷”、“文明的史诗”三大主线，呈现了起源之谜、生命长河、演化之道、大地探珍、缤纷生命、生态万象、生存智慧、人地之缘、上海故事、未来之路等10个常设展区及临展厅、4D影院、探索中心等配套功能区域。')

    print(gpt.generate(user_prompt, system_prompt))
    # GPTCompletion(prompt='结合这段介绍：银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等[4]，其裸露的种子称为白果，叶称蒲扇[5]。介绍银杏。\n', completion='银杏是一种落叶乔木，寿命可达3000年以上，因其形态类似鸭掌而被称为公孙树、鸭掌树、鸭脚树、鸭脚子等。银杏的种子裸露在外，称为白果，叶子则形如蒲扇。银杏是一种非常古老的树种，被誉为“活化石”，在上海自然博物馆的“演化的乐章”展区中可以了解到银杏的演化历程以及其在地球生态系统中的重要角色。银杏还被广泛应用于药用、食用、园林绿化等领域，具有很高的经济价值和文化价值。', prompt_tokens=353, completion_tokens=240)
