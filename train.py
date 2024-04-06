# %%
import multiprocessing, torch, re, tensorboard
import os.path as osp
import datasets

from transformers import EncoderDecoderModel, BertTokenizerFast,\
    Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, BertModel, T5ForConditionalGeneration, T5TokenizerFast, AutoModel, AutoConfig
from tokenization_kobert import KoBertTokenizer
from datasets import load_dataset, Dataset
from evaluate import load as load_metric
from tqdm.auto import tqdm
from time import strftime, time, localtime
from os import listdir
from functools import partial
from typing import List
from zipfile import ZipFile

from utils import rouge, bleu, exact_match, load_dataset, load_iter_augmented_train, load_iter_train_test

print(torch.__version__)
torch.cuda.is_available()

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
# model = EncoderDecoderModel.from_pretrained("kykim/bertshared-kor-base")
config = AutoConfig.from_pretrained("psyche/KoT5-summarization")
model = T5ForConditionalGeneration(config)
# model = T5ForConditionalGeneration.from_pretrained("psyche/KoT5-summarization")
tokenizer = AutoTokenizer.from_pretrained("psyche/KoT5-summarization")
# model = EncoderDecoderModel.from_encoder_decoder_pretrained("kykim/bert-kor-base", "kykim/bert-kor-base")
# tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base")
# model

# %%
tokenizer.eos_token

# %%
tokenizer.bos_token_id

# %%
model.config.decoder_start_token_id

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
tokenizer.model_max_length = 32

# %%
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

# %%
def get_time_dir(): return f"{strftime('%m-%d-%H-%M', localtime(time()))}"

# %%
# dataset = load_dataset(tokenizer=tokenizer)
trainset, testset = load_iter_train_test(tokenizer=tokenizer)

# %%
def batch_iterator():
    dset = iter(load_iter_augmented_train())
    # for i in tqdm(range(0, 1000000, batch_size)):
    for i in tqdm(range(1000000)):
        # yield trainset[i:i+batch_size]['text']
        _item = next(dset)
        # print(_item)
        yield _item['text']
if [_dir for _dir in listdir() if "tokenizer" in _dir] != []:
    latest_tokenizer_path = sorted([_dir for _dir in listdir() if "tokenizer" in _dir])[-1]
    tokenizer = AutoTokenizer.from_pretrained(latest_tokenizer_path)
else:
    # tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2").train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
    # tokenizer = AutoTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator").train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
    tokenizer = AutoTokenizer.from_pretrained("psyche/KoT5-summarization").train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32000)
    tokenizer.save_pretrained("tokenizer_"+get_time_dir())

tokenizer.pad_token = '[PAD]'
tokenizer.eos_token = '[SEP]'
tokenizer.bos_token = '[SEP]'
# tokenizer.unk_token = '[UNK]'
tokenizer.mask_token = '[MASK]'

# %%
tokenizer.encode("안가ㅇ") 

# %%
tokenizer.decode(5206)

# %%
tokenizer.decode(tokenizer.encode("집가"))

# %%
tokenizer("안녀")

# %%
# one = dataset['train'][0]
# one

# %%
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.encoder.vocab_size


# %%
args = Seq2SeqTrainingArguments(
    output_dir=f"output-"+get_time_dir(),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    eval_steps=2_000,
    logging_steps=1,
    gradient_accumulation_steps=8,
    # num_train_epochs=1,
    weight_decay=.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=1e-5,
    save_steps=2_000,
    bf16=True,
    num_train_epochs=5,
    save_total_limit=1,
    max_steps=100_000,
    gradient_checkpointing=True,
    # metric_for_best_model='rouge1_fmeasure',
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics=partial(exact_match, tokenizer=tokenizer),
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


