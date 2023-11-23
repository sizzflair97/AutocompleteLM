import json
import random
from .makeNoisy import split_word
from .makeWord import combine_word
from .recoverWord import recover_word, convert_num
from .language_identifier import contains_english, contains_only_korean
from .exclude_special import exclude_special_characters
from .identify_special_character import get_special_character_index_and_removed_string
from .unicode import join_jamos
from jamo import h2j, j2hcj
from .typos_dict import querty_typos_dict, sgh_typos_dict
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

model_weight_parameter = 100  # 크면 클수록 전체적인 오타가 늘어나는 경향성을 보임

base_keyboard = 'querty'

def make_error(x : str): #하나의 단어에 대해 여러 개의 오타 생성 가능 -> random하게 결정 or 전부 생성
    split_korean_original, result = exclude_special_characters(x)

    index_list, split_korean = get_special_character_index_and_removed_string(split_korean_original)

    try:
        split_korean = split_word(split_korean)[0]
    except:
        if len(split_korean) == 1:
            return typos_generator(split_korean, model_weight_parameter, base_keyboard)
        pass
    split_korean = convert_num(split_korean)

    typos_list = []

    for k in range(1, len(split_korean)):
        recovered = combine_word(recover_word(split_korean[:k]))
        recovered, result = exclude_special_characters(recovered)

        if len(index_list) >= 1:
            for i in range(len(index_list)):
                if index_list[i] <= len(recovered):
                        if recovered[:index_list[i]] == split_korean_original[:index_list[i]]:
                            recovered = recovered[:index_list[i]] + split_korean_original[index_list[i]] + recovered[index_list[i]:]
            typos_text = typos_generator(recovered, model_weight_parameter, base_keyboard)
            typos_list.append(typos_text)
            # print(typos_text)
        
        else:
            result = combine_word(recover_word(split_korean[:k]))
            typos_text = typos_generator(result, model_weight_parameter, base_keyboard)
            typos_list.append(typos_text)
        
            # print(typos_text)
    try:
        random_index = random.randint(0, len(typos_list) - 1)
    except:
        random_index = 0
    return typos_list[random_index]

# make_error("안녕하세요")