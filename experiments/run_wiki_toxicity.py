import csv
import logging
from pathlib import Path

import numpy as np
from scipy.special import softmax
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

    dataset_key = "debias_" if data_args.is_debias else ""
    train_set, valid_set, test_set  = load_dataset("csv",
                                                   data_files={"train": f"data/wiki_toxicity/wiki_{dataset_key}train.csv",
                                                               "validation": f"data/wiki_toxicity/wiki_{dataset_key}dev.csv",
                                                               "test": f"data/wiki_toxicity/wiki_{dataset_key}test.csv"},
                                                   split=["train[:20]", "validation[:200]", "test[:200]"])

    def preprocess(dataset):
        dataset = dataset.map(lambda ex: tokenizer(ex["comment"], padding="max_length", max_length=data_args.max_seq_length, truncation=True), batched=True)
        dataset = dataset.map(lambda ex: {"toxicity": ex["toxicity"] if ex["toxicity"] is not None else 0.0})
        dataset = dataset.map(lambda ex: {"label": ex["is_toxic"]})
        return dataset

    train_set = preprocess(train_set)
    valid_set = preprocess(valid_set)
    test_set = preprocess(test_set)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )

    def compute_metrics(p):
        # take account of only positive probs.
        preds = softmax(p.predictions, axis=-1)[:, 1]

        acc = metrics.accuracy_score(p.label_ids, preds >= 0.5)

        (tn, fp, fn, tp) = metrics.confusion_matrix(p.label_ids, preds >=0.5, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (tn + fn)
        auc = metrics.roc_auc_score(p.label_ids, preds, labels=[0, 1])

        results = {"auc": auc, "accuracy": acc, "false_positive_rate": fpr, "false_negative_rate": fnr}

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

    system_output_dir = Path('runs') / "wiki_toxicity"
    system_output_dir /= model_args.model_name_or_path if model_args.model_name_or_path.startswith("models/") else f"models/{model_args.model_name_or_path}"
    output_file_name = f"epoch_{train_args.num_train_epochs}_lr_{train_args.learning_rate}_batch_{train_args.per_device_train_batch_size}_max_seq_len_{data_args.max_seq_length}"
    if data_args.is_debias:
        output_file_name += "_debiased"
    system_output_dir /= output_file_name

    system_output_dir.mkdir(parents=True, exist_ok=True)

    if train_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path)
        trainer.save_model(str(system_output_dir / 'model'))

    if train_args.do_eval:
        eval_metrics = trainer.evaluate(eval_dataset=valid_set)
        output_eval_metrics_file = system_output_dir / "eval_results.txt"
        if trainer.is_world_process_zero():
            with open(output_eval_metrics_file, "w") as writer:
                for key, value in eval_metrics.items():
                    writer.write(f"{key} = {value}\n")

    if train_args.do_predict:
        output_test_preds_file = system_output_dir / "test_toxicity_predictions.tsv"
        output_test_metrics_file = system_output_dir / "test_results.txt"

        result = trainer.predict(test_dataset=test_set)

        test_metrics = result.metrics
        test_preds = softmax(result.predictions, axis=-1)[:, 1]

        if trainer.is_world_process_zero():
            with open(output_test_metrics_file, "w") as writer:
                for key, value in test_metrics.items():
                    writer.write(f"{key} = {value}\n")

            with open(output_test_preds_file, "w") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["index", "score", "toxicity"])

                for index, score in enumerate(test_preds):

                    writer.writerow([index, score, score >= 0.5])


if __name__ == "__main__":
    run()
