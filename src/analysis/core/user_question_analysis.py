# -*- coding: UTF-8 -*-

def user_question_analysis(question_list):
    """
    分析用户的个人提问数据，统计其所有提问中出现频数最高的标签并返回\n
    返回的标签可以用作问题推荐的参考

    :param question_list: SQL查询得到的当前用户的历史问题列表，包含问题与对应的展品标签
    :return: 按照标签出现频数降序排列的列表
    """
    label_dict = {}
    for i in question_list:
        label = i.label
        cnt = label_dict.get(label)
        if cnt is not None:
            cnt += 1
        else:
            cnt = 1
        label_dict[label] = cnt

    sorted_list = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)

    user_preference = [i[0] for i in sorted_list]

    return user_preference
