import json
from .makeNoisy import split_word
from .makeWord import combine_word
from .recoverWord import recover_word, convert_num
from .language_identifier import contains_english, contains_only_korean
from .exclude_special import exclude_special_characters

y = "안,녕,"

special_character_list = ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', "-", "=", "\\",
                        '~', '!', '@', "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "|",
                        "[", "]", "{", "}",
                        ";", "'", ":", '"',
                        ",", ".", "/",
                        "<", ">", "?"]

def get_special_character_index_and_removed_string(x):

    special_index_list=[]

    for i in range(len(x)):
        if x[i] in special_character_list:
            special_index_list.append(i)

    for j in range(len(x), -1, -1):
        if j in special_index_list:
            x = x[:j] + x[j+1:]

    return special_index_list, x