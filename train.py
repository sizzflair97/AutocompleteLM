# %%
import multiprocessing, torch, re
import os.path as osp
import datasets

from transformers import EncoderDecoderModel, BertTokenizerFast,\
    Seq2SeqTrainer, Seq2SeqTrainingArguments
from tokenization_kobert import KoBertTokenizer
from datasets import load_dataset, Dataset
from evaluate import load as load_metric
from tqdm.auto import tqdm
from time import strftime, time, localtime
from os import listdir
from functools import partial
from typing import List
from zipfile import ZipFile

from utils import rouge, load_dataset

print(torch.__version__)

# %%
# model_path = "skt/kogpt2-base-v2"
# tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, 
#                                                     bos_token='</s>', eos_token='</s>', unk_token='<unk>',
#                                                     pad_token='<pad>', mask_token='<mask>', max_length=512)
# model = GPT2LMHeadModel.from_pretrained(model_path)
# model = BertForMaskedLM.from_pretrained("skt/kobert-base-v1")
# model = AutoModel.from_pretrained('monologg/distilkobert')
# model = BertForMaskedLM.from_pretrained("monologg/kobert-lm")
# tokenizer = KoBertTokenizer.from_pretrained('monologg/distilkobert')
model = EncoderDecoderModel.from_pretrained("kykim/bertshared-kor-base")
tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")
model

# %%
# sent = "안녕하세요.[MASK][MASK][MASK][MASK][MASK][MASK]"

# collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
# encoded = tokenizer(sent, return_tensors="pt")
# target = tokenizer("안녕하세요", return_tensors='pt')

# out = model(**encoded)
# pipe = FillMaskPipeline(model=model, tokenizer=tokenizer)
# while True:
#     t = model(**encoded)
#     logits:torch.Tensor = t.logits
#     predicted_token = logits[:,-1,:].argmax(dim=1)
#     print(tokenizer.decode(predicted_token))

# print(tokenizer.decode(logits[:,-1,:].argmax(dim=1)))

# %%
# pipe("안녕하[MASK][MASK]")

# %%
# mask_idxs = encoded['input_ids'] == tokenizer.mask_token_id
# print(mask_idxs)
# print(out.logits.shape)
# masks = out.logits[mask_idxs].argmax(dim=-1)
# print(masks)
# print(tokenizer.decode(masks))
# predicted_token = out.logits[0,-1].argmax(-1)
# tokenizer.decode(predicted_token)

# %%
# dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
# dataset = load_dataset("wikipedia", "20220301.en")
# dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
# dataset = load_dataset("bookcorpus", split="train", streaming=True).with_format('torch')
# dataset = load_dataset("bookcorpus", split="train")
# dataset

# %%
# def batch_iterator(batch_size=10000):
#     for i in tqdm(range(0, len(dataset['train']), batch_size)):
#         yield dataset['train'][i:i+batch_size]['text']
# if [_dir for _dir in listdir() if "tokenizer" in _dir] != []:
#     latest_tokenizer_path = sorted([_dir for _dir in listdir() if "tokenizer" in _dir])[-1]
#     tokenizer = AutoTokenizer.from_pretrained(latest_tokenizer_path)
# else:
#     # tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2").train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
#     tokenizer = AutoTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator").train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
#     tokenizer.save_pretrained(get_time_dir())

# tokenizer.pad_token = '<pad>'
# tokenizer.eos_token = '</s>'
# tokenizer.bos_token = '</s>'
# tokenizer.unk_token = '<unk>'
# tokenizer.mask_token = '<mask>'
# tokenizer.model_max_length = 32

# %%
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

# %%
def get_time_dir(): return f"{strftime('%m-%d-%H-%M', localtime(time()))}"

# %%
from utils import load_iter_train_test
# dataset = load_dataset(tokenizer=tokenizer)
trainset, testset = load_iter_train_test(tokenizer=tokenizer)

# %%
# one = dataset['train'][0]
# one

# %%
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.config.max_length = 32
model.config.early_stopping = True
# model.config.no_repeat_ngram_size = 1
model.config.length_penalty = 2.
model.config.repetition_penalty = 3.
model.config.num_beams = 10
model.config.vocab_size = model.config.encoder.vocab_size


# %%
args = Seq2SeqTrainingArguments(
    output_dir=f"output-{get_time_dir()}",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    predict_with_generate=True,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    eval_steps=1_000,
    logging_steps=1_000,
    gradient_accumulation_steps=8,
    # num_train_epochs=1,
    weight_decay=.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=2_000,
    fp16=True,
    num_train_epochs=5,
    save_total_limit=1,
    max_steps=100_000,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics=partial(rouge, tokenizer=tokenizer),
    train_dataset=trainset,
    eval_dataset=testset,
)

# %%
# config = AutoConfig.from_pretrained(
#     "gpt2",
#     vocab_size = len(tokenizer),
#     n_ctx = tokenizer.model_max_length,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
# )
# model = GPT2LMHeadModel(config)
# model_size = sum(t.numel() for t in model.parameters())
# print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %%
trainer.train()

# %%



