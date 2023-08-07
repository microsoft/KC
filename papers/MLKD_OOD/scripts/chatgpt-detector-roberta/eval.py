# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm
from utils.ood_metrics import compute_all_scores
import joblib


def prediction_classification(path, model, tokenizer):
    bsz = 8
    with open(path) as f:
        texts = [ii.strip() for ii in f.readlines()]
    res = []
    for ii in tqdm.tqdm(range(0, len(texts), bsz)):
        batch = texts[ii:ii + bsz]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {ii: jj.to("cuda:0") for ii, jj in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        res.extend([ii[0] for ii in logits])
    return res

def main():
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta").to("cuda:0")
    model.eval()
    iid = prediction_classification("OOD_dataset/HC3/hc3_test.txt", model, tokenizer)
    ood = prediction_classification("OOD_dataset/HC3/hc3_ood.txt", model, tokenizer)
    joblib.dump([iid, ood], "x.pkl")
    res = compute_all_scores(id_scores=iid, ood_scores=ood, output_dir="x")

if __name__ == "__main__":
    main()