import random
from unicode import join_jamos
from jamo import h2j, j2hcj
from typos_dict import querty_typos_dict, sgh_typos_dict
import numpy as np


def total_prob_weight_function(len_text, model_weight_parameter):
    weighted_prob = np.log(len_text+1)/model_weight_parameter+1
    return weighted_prob


def each_probability_weight(weighted_prob, typos_dict):
    prob_sum = 0
    transformed_prob_distribution = []
    for _, prob in typos_dict:
        typing_prob = prob*weighted_prob
        prob_sum += typing_prob
        transformed_prob_distribution.append(typing_prob)
    # print("prob_sum: ", prob_sum)
    return prob_sum, transformed_prob_distribution


def typos_generator(text, model_weight_parameter, base_keyboard):
    weighted_prob = total_prob_weight_function(
        len(text), model_weight_parameter)
    jamo_text = j2hcj(h2j(text))
    # print(f"확률 가중치: {weighted_prob:.7f}")
    typos_str = ''
    for jamo in jamo_text:
        if base_keyboard == 'querty':
            typos_dict = querty_typos_dict[jamo]
        elif base_keyboard == '천지인':
            typos_dict = sgh_typos_dict[jamo]
        # print("="*50)
        probability = 0
        max_parameter, transformed_prob_distribution = each_probability_weight(
            weighted_prob, typos_dict)
        select = random.uniform(
            0, max_parameter)
        # print(typos_dict[jamo])
        # print("transformed_prob_distribution \n",
        #      transformed_prob_distribution)
        stack = 0
        for char, _ in typos_dict:
            probability += transformed_prob_distribution[stack]
            stack += 1
            if select <= probability:
                typos_str = typos_str + char
                # print("stack: ", stack)
                # print(f"select: {select:.7f}, 선택된 문자: {char}")
                break
    # 변형 발생 여부 확인 코드 작성
    return join_jamos(typos_str)


input_dictionary = {'data': ["지구", "웨의", "모든", "일웨ㅇ"], 'target': "일은"}

model_weight_parameter = 100  # 크면 클수록 전체적인 오타가 늘어나는 경향성을 보임

base_keyboard = 'querty'
"""
querty keyboard: 기본 자판
천지인 keyboard: 구형 자판
"""
text = input_dictionary["data"][-1]
print("input_dictionary ", input_dictionary)
typos_text = typos_generator(text, model_weight_parameter, base_keyboard)
input_dictionary["data"][-1] = typos_text
print("output_dictionary ", input_dictionary)