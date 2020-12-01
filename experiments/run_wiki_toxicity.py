from comet_ml import Experiment

import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import HfArgumentParser, TrainingArguments, Trainer, default_data_collator, set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn import metrics
from datasets import load_dataset
from dotenv import load_dotenv

from arguments import ModelArguments, SeqClassificationDataArguments

ARGS_JSON_FILE = "args_wiki_toxicity.json"
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
            model_args.config_name, cache_dir=model_args.cache_dir, num_labels=1
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, num_labels=1
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

    train_set, valid_set, test_set  = load_dataset("csv",
                                                   data_files={"train": "data/wiki_toxicity/wiki_debias_train.csv",
                                                               "validation": "data/wiki_toxicity/wiki_debias_dev.csv",
                                                               "test": "data/wiki_toxicity/wiki_debias_test.csv"},
                                                   split=["train", "validation", "test"])

    def preprocess(dataset):
        dataset = dataset.map(lambda ex: tokenizer(ex["comment"], padding="max_length", max_length=data_args.max_seq_length, truncation=True), batched=True)
        dataset = dataset.map(lambda ex: {"toxicity": ex["toxicity"] if ex["toxicity"] is not None else 0.0})
        dataset = dataset.map(lambda ex: {"label": float(ex["is_toxic"])})
        return dataset

    train_set = preprocess(train_set)
    valid_set = preprocess(valid_set)
    test_set = preprocess(test_set)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )

    def compute_metrics(p):
        preds = p.predictions
        preds = np.argmax(preds, axis=1)

        print(p.label_ids)
        fpr, tpr, thresholds = metrics.roc_curve(p.label_ids, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        results = {"auc": auc, "false_positive_rate": fpr, "true_positive_rate": tpr, "thresholds": thresholds}
        return results

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator
    )

    if train_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path)
        trainer.save_model(train_args.logging_dir)

    system_output_dir = Path('runs') / "wiki_toxicity"
    system_output_dir /= model_args.model_name_or_path if model_args.model_name_or_path.startswith("models/") else f"models/{model_args.model_name_or_path}"
    system_output_dir /= f"epoch_{train_args.num_train_epochs}_lr_{train_args.learning_rate}_batch_{train_args.per_device_train_batch_size}_max_seq_len_{data_args.max_seq_length}"

    system_output_dir.mkdir(parents=True, exist_ok=True)

    if train_args.do_eval:
        eval_metrics = trainer.evaluate(eval_dataset=valid_set)
        output_eval_metrics_file = system_output_dir / "eval_results.txt"
        if trainer.is_world_process_zero():
            with open(output_eval_metrics_file, "w") as writer:
                for key, value in eval_metrics.items():
                    writer.write(f"{key} = {value}\n")

    if train_args.do_predict:
        output_test_preds_file = system_output_dir / "test_toxicity_predictions.tsv"
        output_test_metrics_file = system_output_dir / "eval_results.txt"

        result = trainer.predict(test_dataset=test_set)

        test_metrics = result.metrics
        test_preds = np.squeeze(result.predictions)

        if trainer.is_world_process_zero():
            with open(output_test_metrics_file, "w") as writer:
                for key, value in test_metrics.items():
                    writer.write(f"{key} = {value}\n")

            with open(output_test_preds_file, "w") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["index", "toxicity"])

                for index, toxicity in enumerate(test_preds):

                    writer.writerow([index, toxicity])


if __name__ == "__main__":
    run()
