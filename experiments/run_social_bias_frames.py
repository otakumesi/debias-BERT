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

ARGS_JSON_FILE = "args_social_bias_frames.json"
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

    train_set, valid_set, test_set = load_dataset("social_bias_frames", split=["train[:20]", "validation[:20]", "test[:20]"])

    with open('data/masking_words.txt') as f:
        masking_words = [line.strip() for line in f.readlines()]

    def transform_word_to_mask(exs):
        sentences = []
        for i, data in enumerate(exs["post"]):
            text = data[0]
            results = predictor.predict(sentence=text)
            words = [results["words"][i] for i, tag in enumerate(results["tags"]) if any([tag.endswith(ent_name) for ent_name in ["PER", "LOC", "MISC"]])]
            sentence = text
            for word in words:
                sentence = re.sub(re.escape(word), "[MASK]", sentence, flags=re.IGNORECASE)
            for word in masking_words:
                sentence = re.sub(rf"\b({word}|{word}'s)\b", "[MASK]", sentence, flags=re.IGNORECASE)
        sentences.append(sentence)
        return {"sentences": sentences}

    def preprocess(dataset):
        if data_args.is_masking_words:
            dataset = dataset.map(transform_word_to_mask, batched=True)
        dataset = dataset.map(ex: {"label": ex["offensiveYN"]})
        dataset = dataset.map(ex: {"targetMinoriy": ex["targetMinoriy"] if ex["targetMinority"] is not None else "others"})
        dataset = dataset.map(ex: {"targetCategory": ex["targetCategory"] if ex["targetCategory"] is not None else "others"})
        return dataset

    train_set = preprocess(train_set)
    valid_set = preprocess(valid_set)
    test_set = preprocess(test_set)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )


    def compute_bias_metrics(preds, overall_metrics, dataset):
        df = pd.DataFrame({"preds": preds, "labels": dataset["label"], "subgroup": dataset["targetMinority"]})
        subgroups = set(df["subgroup"])

        overall_auc = overall_metrics["auc"]
        overall_fpr = overall_metrics["false_positive_rate"]
        overall_fnr = overall_metrics["false_negative_rate"]

        result_metrics = {"pinned_auc": 0., "false_positive_equality_diff": 0., "false_negative_equality_diff": 0.}
        for subgroup in subgroups:
            sub_df = df[df["subgroup"] == subgroup]
            non_sub_df = df[df["subgroup"] != subgroup].sample(len(sub_df), random_state=train_args.seed)
            pinned_df = pd.concat([sub_df, non_sub_df])
            (auc, fpr, fnr) = compute_primitive_metrics(sub_df["labels"], sub_df["preds"])
            result_metrics[f"{term}_auc"] = auc
            result_metrics[f"{term}_fpr"] = fpr
            result_metrics[f"{term}_fnr"] = fnr
            result_metrics["pinned_auc"] += np.abs(overall_auc - metrics.roc_auc_score(pinned_df["labels"], pinned_df["preds"], labels=[0, 1]))
            result_metrics["false_positive_equality_diff"] += np.abs(overall_fpr - fpr)
            result_metrics["false_negative_equality_diff"] += np.abs(overall_fnr - fnr)
        return result_metrics


    def compute_metrics(p):
        # take account of only positive probs.
        preds = softmax(p.predictions, axis=-1)[:, 1]

        acc = metrics.accuracy_score(p.label_ids, preds >= 0.5)

        (tn, fp, fn, tp) = metrics.confusion_matrix(p.label_ids, preds >=0.5, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (tn + fn)
        auc = metrics.roc_auc_score(p.label_ids, preds, labels=[0, 1])

        results = {"auc": auc, "accuracy": acc, "false_positive_rate": fpr, "false_negative_rate": fnr}

        results = results.merge(compute_bias_metrics(preds, results, dataset))

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

    system_output_dir = Path('runs') / "social_bias_frames"
    system_output_dir /= model_args.model_name_or_path if model_args.model_name_or_path.startswith("models/") else f"models/{model_args.model_name_or_path}"
    output_file_name = f"epoch_{train_args.num_train_epochs}_lr_{train_args.learning_rate}_batch_{train_args.per_device_train_batch_size}_max_seq_len_{data_args.max_seq_length}"
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
        output_test_preds_file = system_output_dir / "test_offensive_predictions.tsv"
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
                writer.writerow(["index", "score", "offensive"])

                for index, score in enumerate(test_preds):

                    writer.writerow([index, score, score >= 0.5])


if __name__ == "__main__":
    run()
