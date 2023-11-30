from evaluate import load
from transformers import BertTokenizerFast

rouge_module = load("rouge")

def rouge(pred, tokenizer:BertTokenizerFast, rouge_module=rouge_module):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # rouge_output = rouge_module.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    # return {
    #     "rouge2_precision": round(rouge_output.precision, 4),
    #     "rouge2_recall": round(rouge_output.recall, 4),
    #     "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    # }
    rouge_output = rouge_module.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
    # print(rouge_output)
    return rouge_output