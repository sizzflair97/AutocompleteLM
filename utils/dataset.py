import datasets, random, re
import os.path as osp

from datasets import Dataset, IterableDataset
from transformers import BertTokenizerFast
from typing import List, Tuple
from functools import partial
from zipfile import ZipFile
from multiprocessing import cpu_count
from collections import defaultdict
from random import sample

from .korean_english_multitarget_ted_talks_task.main_func import make_error
from .korean_english_multitarget_ted_talks_task.exclude_special import exclude_special_characters
from .korean_english_multitarget_ted_talks_task.language_identifier import contains_only_korean

def cut_start_if_1_or_2(text):
    if text.startswith("1 :"):
        return text[3:]  # Cut the string starting from the third character
    elif text.startswith("2 :"):
        return text[3:]  # Cut the string starting from the third character
    else:
        return text
    
def make_four_words(batch_text):
    batch_text = batch_text.split("\n")
    flag = 0
    # print(batch_text)
    while True: #모든 text가 4이하일 때를 고려해야 할 듯
        flag += 1
        rand_index = random.randint(0, len(batch_text) - 1)
        batch_text_ex = batch_text[rand_index]
        batch_text_ex = exclude_special_characters(cut_start_if_1_or_2(batch_text_ex))[0].split()
        if len(batch_text_ex) >= 4:
            break
        if flag >= 100: #4이상인 text를 찾을 수 없는 경우
            return False
    random_idx = random.randint(0, len(batch_text_ex) -4)#총 4개의 토큰 -> 총 가능한 경우의 수 : len - 3
    batch_text = batch_text_ex[random_idx:random_idx + 4]
    if not contains_only_korean(batch_text[-1]):
        return False

    return batch_text

def make_label(batch_text):
    result_list = []
    for i in range(len(batch_text['text'])):
        result = make_error(batch_text['text'][i][-1])
        result_list.append(result)

    return result_list


def make_error_batch_text(batch_text):
    for i in range(len(batch_text)):
        # print("error word : ", batch_text[i][-1])
        batch_text[i][-1] = make_error(batch_text[i][-1])
        batch_text[i]= [' '.join(batch_text[i])][0]

    return batch_text

def augment_text(batch:dict):
    for i, s in enumerate(batch['text']):
        for match in reversed([*re.finditer("^[0-9]* : ", s)]):
            s = s[:match.start()] + s[match.end():]
        batch['text'][i] = make_four_words(s)
    batch['text'] = [batch_text for batch_text in batch['text'] if batch_text is not False]
    batch["text"] = make_error_batch_text(batch["text"])    
    return batch

def preprocess_text(batch:dict, tokenizer:BertTokenizerFast):
    for i, s in enumerate(batch['text']):
        for match in reversed([*re.finditer("^[0-9]* : ", s)]):
            s = s[:match.start()] + s[match.end():]
        batch['text'][i] = make_four_words(s)
    batch['text'] = [batch_text for batch_text in batch['text'] if batch_text is not False]
    label:List[str] = [x[-1] for x in batch["text"]]
    
    batch["text"] = make_error_batch_text(batch["text"])
    batch["text"] = ["".join(batch_text) for batch_text in batch['text'] if batch_text]
    
    print(f"text: {batch['text'][0]}, label: {label[0]}")
    
    tokenized_inputs = tokenizer(
        batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
    )
    tokenized_outputs = tokenizer(
        label, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
    )
    
    batch['input_ids'] = tokenized_inputs.input_ids
    batch['attention_mask'] = tokenized_inputs.attention_mask
    # batch['decoder_input_ids'] = tokenized_outputs.input_ids
    # batch['decoder_attention_mask'] = tokenized_outputs.attention_mask
    batch['labels'] = tokenized_outputs.input_ids.copy()
    
    batch['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    
    return batch

def load_dataset(tokenizer:BertTokenizerFast):
    if osp.exists("cache"):
        mapped = datasets.load_from_disk('cache')
    else:
        with ZipFile("data.zip", mode='r') as zf:
            dataset = Dataset.from_dict(
                {"text":[zf.read(fname).decode('UTF-8') for fname in zf.filelist]}
        )
        dataset = dataset.train_test_split(0.3)
        # mapped = dataset.map(partial(preprocess_text, tokenizer=tokenizer), batched=True, remove_columns=['text'], num_proc=cpu_count()) # bug on windows
        mapped = dataset.map(partial(preprocess_text, tokenizer=tokenizer), batched=True, remove_columns=['text'])
        mapped.set_format("torch")
        mapped.save_to_disk("cache")
    return mapped

def text_loader(is_test:bool=False):
    with ZipFile("data.zip", mode='r') as zf:
        flist = sample(zf.filelist, 5000) if is_test else zf.filelist
        while True:
            for fname in flist:
                yield {"text":zf.read(fname).decode('UTF-8')}
            if is_test:
                break

def load_iter_augmented_train() -> Tuple[IterableDataset, IterableDataset]:
    train = IterableDataset.from_generator(text_loader, gen_kwargs={"is_test":False})
    train_mapped = train.map(augment_text, batched=True)
    return train_mapped
                
def load_iter_train_test(tokenizer:BertTokenizerFast) -> Tuple[IterableDataset, IterableDataset]:
    train = IterableDataset.from_generator(text_loader, gen_kwargs={"is_test":False})
    test = IterableDataset.from_generator(text_loader, gen_kwargs={"is_test":True})
    train_mapped = train.map(partial(preprocess_text, tokenizer=tokenizer), batched=True, remove_columns=['text'])
    test_mapped = test.map(partial(preprocess_text, tokenizer=tokenizer), batched=True, remove_columns=['text'])
    # mapped.set_format("torch")
    # mapped.save_to_disk("cache")
    return train_mapped, test_mapped
