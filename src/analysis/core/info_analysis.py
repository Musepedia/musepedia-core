# -*- coding: UTF-8 -*-

def info_analysis(query_list, K):
    """
    针对用户的个人信息，统计不同年龄段的用户分别感兴趣的内容

    :param query_list: SQL的查询结果，内容包括年龄及其对应的问题标签
    :param K: 每个年龄段选取的感兴趣标签的个数
    :return: 各年龄段最感兴趣的TopK个展品标签
    """

    # key为年龄段, value同样采用dict的形式, 记录每个label的出现频数
    age_dict = {}
    # 统计
    for query in query_list:
        age = query.age
        label = query.label
        label_dict = age_dict.get(age)
        if label_dict is not None:
            freq = label_dict.get(label)
            if freq is not None:
                freq += 1
            else:
                freq = 1
            label_dict[label] = freq
        else:
            age_dict[age] = {label: 1}

    print(age_dict)

    # 排序
    for key, value in age_dict.items():
        sorted_list = sorted(value.items(), key=lambda x: x[1], reverse=True)
        age_dict[key] = sorted_list

    # 返回
    for key, value in age_dict.items():
        if len(value) > K:
            age_dict[key] = [i[0] for i in value[:K]]
        else:
            age_dict[key] = [i[0] for i in value]

    return age_dict
