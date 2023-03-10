from loguru import logger


def concat_logging_info(completion):
    return "提示词：{0} ({1} tokens)\t生成回答：{2} ({3} tokens)".format(completion.prompt,
                                                                       completion.prompt_tokens,
                                                                       completion.completion,
                                                                       completion.completion_tokens)


def gpt_logging(run):
    """
    打印与GPT模型相关的Debug级别日志信息，包括传入模型的提示词（prompt）与token数、生成的回答（completion)与token数
    """

    def wrapper(*args, **kwargs):
        result = run(*args, **kwargs)
        logger.debug(concat_logging_info(result))
        return result

    return wrapper


if __name__ == '__main__':
    @gpt_logging
    def get_completion():
        completion = GPTCompletion(
            prompt='结合这段介绍：银杏（学名：Ginkgo biloba），落叶乔木，寿命可达3000年以上。又名公孙树、鸭掌树、鸭脚树、鸭脚子等[4]，其裸露的种子称为白果，叶称蒲扇[5]。介绍银杏。\n',
            completion='银杏是一种落叶乔木，寿命可达3000年以上，因其形态类似鸭掌而被称为公孙树、鸭掌树、鸭脚树、鸭脚子等。银杏的种子裸露在外，称为白果，叶子则形如蒲扇。银杏是一种非常古老的树种，被誉为“活化石”，在上海自然博物馆的“演化的乐章”展区中可以了解到银杏的演化历程以及其在地球生态系统中的重要角色。银杏还被广泛应用于药用、食用、园林绿化等领域，具有很高的经济价值和文化价值。',
            prompt_tokens=353, completion_tokens=240)
        return completion


    get_completion()
