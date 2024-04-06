from evaluate import load
from transformers import BertTokenizerFast

rouge_module = load("evaluate-metric/rouge")
bleu_module = load("evaluate-metric/bleu")
exact_module = load("evaluate-metric/exact_match")

def rouge(pred, tokenizer:BertTokenizerFast, rouge_module=rouge_module):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    output = rouge_module.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])
    
    print(f"pred:{pred_str}, label:{label_str}, score:{str(output)}")
    
    # prediction_lens = [np.count_nonzero(_pred != tokenizer.pad_token_id) for _pred in pred_ids]
    # result["gen_len"] = np.mean(prediction_lens)
    
    # print(rouge_output)
    return output

def bleu(pred, tokenizer:BertTokenizerFast, bleu_module=bleu_module):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    output = bleu_module.compute(predictions=pred_str, references=label_str)
    
    print(f"pred:{pred_str}, label:{label_str}, score:{str(output)}")
    
    # prediction_lens = [np.count_nonzero(_pred != tokenizer.pad_token_id) for _pred in pred_ids]
    # result["gen_len"] = np.mean(prediction_lens)
    
    # print(output)
    return output

def exact_match(pred, tokenizer:BertTokenizerFast, exact_module=exact_module):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    output = exact_module.compute(predictions=pred_str, references=label_str)
    
    print(f"pred:{pred_str}, label:{label_str}, score:{str(output)}")
    
    # prediction_lens = [np.count_nonzero(_pred != tokenizer.pad_token_id) for _pred in pred_ids]
    # result["gen_len"] = np.mean(prediction_lens)
    
    # print(output)
    return output