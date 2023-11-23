import json
import random
from makeNoisy import split_word
from makeWord import combine_word
from recoverWord import recover_word, convert_num
from language_identifier import contains_english, contains_only_korean
from exclude_special import exclude_special_characters
from identify_special_character import get_special_character_index_and_removed_string
from unicode import join_jamos
from jamo import h2j, j2hcj
from typos_dict import querty_typos_dict, sgh_typos_dict
import numpy as np

# Specify the path to your JSON file
json_file_path = 'korean_english_multitarget_ted_talks_task/train.json'

# Open the file in read mode
json_data = []
with open(json_file_path, 'r', encoding='UTF8') as f:
    for line in f:
        json_data.append(json.loads(line))


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

for i in range(len(json_data)):
    token_list = json_data[i]["korean"].split()
    
    #for test
    #token_list = ["안녕", "나는", "행복한", "한,화,팬/"]

    if len(token_list) >= 4:
        data_list = []
        for j in range(3, len(token_list)):
            new_data = {"data" : token_list[j - 3 : j], "label" : token_list[j]}

            if contains_english(new_data['label']): #label단어가 영어인 경우
                continue
                for k in range(1, len(new_data["label"])):
                    last_token_list = token_list[j - 3 : j] + [new_data["label"][:k]]
                    data_list.append({"data" : last_token_list, "label" : token_list[j]})
                    total_num +=1
            else: #label 단어가 한국어인 경우
                split_korean_original = new_data["label"]
                split_korean_original, result = exclude_special_characters(split_korean_original)
                if result == False:
                    continue

                index_list, split_korean = get_special_character_index_and_removed_string(split_korean_original)
                if not contains_only_korean(split_korean):
                    continue
                split_korean = split_word(split_korean)[0]
                split_korean = convert_num(split_korean)
                for k in range(1, len(split_korean)):
                    recovered = combine_word(recover_word(split_korean[:k]))
                    recovered, result = exclude_special_characters(recovered)
                    # print(recovered)
                    # if result == False: #해당 부분은 현우가 수정하면 삭제
                    #     continue
                    # print()

                    if len(index_list) >= 1:
                        for i in range(len(index_list)):
                            if index_list[i] <= len(recovered):
                                if recovered[:index_list[i]] == split_korean_original[:index_list[i]]:
                                    recovered = recovered[:index_list[i]] + split_korean_original[index_list[i]] + recovered[index_list[i]:]
                            last_token_list = token_list[j - 3 : j] + [recovered]
                        new_token = {"data" : last_token_list, "label" : token_list[j]}
                        typos_text = typos_generator(new_token["data"][-1], model_weight_parameter, base_keyboard)
                        new_token["data"][-1] = typos_text
                        data_list.append({"data" : last_token_list, "label" : token_list[j]})
                        # print({"data" : last_token_list, "label" : token_list[j]})
                    else:
                        last_token_list = token_list[j - 3 : j] + [combine_word(recover_word(split_korean[:k]))]
                        new_token = {"data" : last_token_list, "label" : token_list[j]}
                        typos_text = typos_generator(new_token["data"][-1], model_weight_parameter, base_keyboard)
                        new_token["data"][-1] = typos_text
                        data_list.append(new_token)    

                    # last_token_list = token_list[j - 3 : j] + [combine_word(recover_word(split_korean[:k]))]
                    # data_list.append({"data" : last_token_list, "label" : token_list[j]})    
                    # total_num += 1
        
    for data in data_list:
        print(data)
