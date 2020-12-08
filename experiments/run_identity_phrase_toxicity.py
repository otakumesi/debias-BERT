import re
import csv
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import softmax
from transformers import HfArgumentParser, TrainingArguments, Trainer, default_data_collator, set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn import metrics
from datasets import load_dataset
from dotenv import load_dotenv

from arguments import ModelArguments, SeqClassificationDataArguments

ARGS_JSON_FILE = "args_identity_phrase.json"
logging.basicConfig(level=logging.INFO)
load_dotenv()


def run():
    parser = HfArgumentParser(
        (ModelArguments, SeqClassificationDataArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_json_file(ARGS_JSON_FILE)

    set_seed(train_args.seed)

    # Load Transformers config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir, num_labels=2
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, num_labels=2
        )

    # Load Transformers Tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenzier_name, cache_dir=model_args.cache_dir, use_fast=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=True
        )

    (test_set,) = load_dataset("csv",
                             data_files={"train": "data/wiki_toxicity/bias_madlibs_89k.csv"},
                             split=["train"])

    with open("data/wiki_toxicity/adjectives_people.txt") as f:
        terms = [line.strip() for line in f.readlines()]

    def add_subgroups(ex):
        return {"subgroup": [term for term in terms if bool(re.search("\\b" + term + "\\b", ex["Text"], flags=re.UNICODE | re.IGNORECASE))][0]}

    def preprocess(dataset):
        dataset = dataset.map(lambda ex: tokenizer(ex["Text"], padding="max_length", max_length=data_args.max_seq_length, truncation=True), batched=True)
        dataset = dataset.map(lambda ex: {"label": ex["Label"] == "BAD"})
        dataset = dataset.map(add_subgroups)
        return dataset

    test_set = preprocess(test_set)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )

    def compute_primitive_metrics(label_ids, preds):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true=label_ids, y_pred=preds >=0.5, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr = fn / (tn + fn) if (tn + fn) != 0 else 0
        auc = metrics.roc_auc_score(label_ids, preds, labels=[0, 1])

        return (auc, fpr, fnr)


    def compute_metrics(p):
        # take account of only positive probs.
        preds = softmax(p.predictions, axis=-1)[:, 1]
        acc = metrics.accuracy_score(p.label_ids, preds >= 0.5)
        (auc, fpr, fnr) = compute_primitive_metrics(p.label_ids, preds)

        results = {"auc": auc, "accuracy": acc, "false_positive_rate": fpr, "false_negative_rate": fnr}

        return results

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator
    )

    result = trainer.predict(test_dataset=test_set)

    test_metrics = result.metrics
    test_preds = softmax(result.predictions, axis=-1)[:, 1]

    def compute_bias_metrics(preds, overall_metrics, dataset):
        df = pd.DataFrame({"preds": preds, "labels": dataset["label"], "subgroup": dataset["subgroup"]})

        overall_auc = overall_metrics["eval_auc"]
        overall_fpr = overall_metrics["eval_false_positive_rate"]
        overall_fnr = overall_metrics["eval_false_negative_rate"]

        result_metrics = {"pinned_auc": 0., "false_positive_equality_diff": 0., "false_negative_equality_diff": 0.}
        for term in terms:
            sub_df = df[df["subgroup"] == term]
            if len(sub_df) == 0:
                continue
            non_sub_df = df[df["subgroup"] != term].sample(len(sub_df), random_state=train_args.seed)
            pinned_df = pd.concat([sub_df, non_sub_df])
            (auc, fpr, fnr) = compute_primitive_metrics(sub_df["labels"], sub_df["preds"])
            result_metrics[f"{term}_auc"] = auc
            result_metrics[f"{term}_fpr"] = fpr
            result_metrics[f"{term}_fnr"] = fnr
            result_metrics["pinned_auc"] += np.abs(overall_auc - metrics.roc_auc_score(pinned_df["labels"], pinned_df["preds"], labels=[0, 1]))
            result_metrics["false_positive_equality_diff"] += np.abs(overall_fpr - fpr)
            result_metrics["false_negative_equality_diff"] += np.abs(overall_fnr - fnr)
        return result_metrics

    bias_metrics = compute_bias_metrics(test_preds, test_metrics, test_set)

    system_output_dir = Path(model_args.model_name_or_path).parent
    output_test_preds_file = system_output_dir / "test_identity_toxicity_predictions.tsv"
    output_test_metrics_file = system_output_dir / "test_identity_results.txt"

    if trainer.is_world_process_zero():
        with open(output_test_preds_file, "w") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["index", "score", "toxicity"])
            for index, score in enumerate(test_preds):
                writer.writerow([index, score, score >= 0.5])

        with open(output_test_metrics_file, "w") as writer:
            for key, value in test_metrics.items():
                writer.write(f"{key} = {value}\n")
            for key, value in bias_metrics.items():
                writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    run()
