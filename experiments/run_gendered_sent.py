from comet_ml import Experiment

import csv
import logging
import random
from pathlib import Path

import numpy as np
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset, load_metric
from dotenv import load_dotenv

from arguments import ModelArguments, GenderedSentimentDataArguments

ARGS_JSON_FILE = "args_gendered_sent.json"
logging.basicConfig(level=logging.INFO)
load_dotenv()


def run():
    parser = HfArgumentParser(
        (ModelArguments, GenderedSentimentDataArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_json_file(ARGS_JSON_FILE)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # Load Transformers config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
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

    train_set, eval_set = load_dataset(
        "glue", "sst2", split=["train", "validation"]
    )
    test_set = load_dataset(
        "csv", "sst2", data_files="data/gendered-sentiment/test.tsv", delimiter="\t", split="train"
    )
    metric = load_metric("glue", "sst2")

    def preprocess(examples):
        return tokenizer(examples["sentence"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

    train_set = train_set.map(preprocess, batched=True)
    eval_set = eval_set.map(preprocess, batched=True)
    test_set = test_set.map(preprocess, batched=True)


    columns = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "label"
    ]

    train_set.set_format(type="torch", columns=columns)
    eval_set.set_format(type="torch", columns=columns)
    test_set.set_format(type="torch", columns=columns[:-1])


    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )

    def compute_metrics(p):
        preds = p.predictions
        preds = np.argmax(preds)

        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics
    )

    if train_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path)
        trainer.save_model(train_args.logging_dir)

    system_output_dir = Path('runs') / 'models' / model_args.model_name_or_path / \
            f"epoch_{train_args.epoch}_lr_{train_args.learning_rate}_batch_{train_args.per_device_train_batch_size}"
    system_output_dir.mkdir(parents=True, exist_ok=True)

    if train_args.do_eval:
        if data_args.task_name == "mnli":
            eval_result = trainer.evaluate(eval_dataset=eval_set)
            output_eval_file = system_output_dir / "eval_results.txt"
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    for key, value in eval_result.items():
                        writer.write(f"{key} = {value}\n")

    if train_args.do_predict:
        system_output = system_output_dir / "gendered_sent_predicts.tsv"

        preds = trainer.predict(test_dataset=test_set).predictions
        preds = np.squeeze(preds)

        if trainer.is_world_process_zero():
            with open(system_output, "w") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["index", "prediction_0", "prediction_1"])

                for index, (label_0, label_1) in enumerate(preds):

                    writer.writerow([index, label_0, label_1])


if __name__ == "__main__":
    run()
