# -*- coding: UTF-8 -*-

def question_analysis(query_list, K):
    """
    分析历史提问，将查询数据库的结果进行整理后返回

    :param query_list: sql返回的多条查询结果，(问题，问题的频数，展品标签）
    :param K: 返回的问题个数及展品个数
    :return: 出现次数最多的K个问题、该K个问题的频数、出现次数最高的K个展品、该K个展品的频数
    """

    questions = [i.question for i in query_list]
    exhibit_labels = [i.exhibitLabel for i in query_list]
    freq = [i.freq for i in query_list]

    # 判断当前的所有数据是否足够K条，不够则全部返回
    length = len(questions)
    if length < K:
        K = length

    # 使用字典统计各id的出现次数
    id_dict = {}
    for i in range(length):
        id = exhibit_labels[i]
        cnt = id_dict.get(id)
        question_cnt = freq[i]
        if cnt is not None:
            cnt += question_cnt
            id_dict[id] = cnt
        else:
            id_dict[id] = question_cnt

    # 按照字典的value进行排序
    sorted_id = sorted(id_dict.items(), key=lambda x: x[1], reverse=True)

    # 返回列表
    reply_questions = questions[:K]
    reply_freq = freq[:K]
    reply_labels = [i[0] for i in sorted_id[:K]]
    reply_label_freq = [i[1] for i in sorted_id[:K]]
    return reply_questions, reply_freq, reply_labels, reply_label_freq

