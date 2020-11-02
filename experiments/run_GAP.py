from comet_ml import Experiment

import os
import csv
import logging
import random
from pathlib import Path

import numpy as np
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from datasets import load_dataset

from arguments import ModelArguments, WinobiasDataArguments
from models import MyCorefResolver
from utils import prepare_gap

ARGS_JSON_FILE = "args.json"
logging.basicConfig(level=logging.INFO)

load_dotenv()

API_KEY = os.getenv("COMET_API_KEY")
WORKSPACE = os.getenv("COMET_WORKSPACE")
PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")

experiment = Experiment(
    api_key=API_KEY,
    workspace=WORKSPACE,
    project_name=PROJECT_NAME,
    auto_output_logging="simple",
)


def run():
    parser = HfArgumentParser(
        (ModelArguments, WinobiasDataArguments, TrainingArguments)
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

    train_set, valid_set, test_set = load_dataset(
        "gap", split=["train", "validation", "test"]
    )

    # For Output files
    test_set_ids = test_set["ID"].copy()

    train_set = prepare_gap(train_set, tokenizer).shuffle(seed=train_args.seed)
    valid_set = prepare_gap(valid_set, tokenizer)
    test_set = prepare_gap(test_set, tokenizer)

    columns = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "a_span_indeces",
        "b_span_indeces",
        "p_span_indeces",
        "labels",
    ]
    train_set.set_format(type="torch", columns=columns)
    valid_set.set_format(type="torch", columns=columns)
    test_set.set_format(type="torch", columns=columns)

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )
    resolver = MyCorefResolver(model=model, dropout=0.8)

    # Fixed pretrained model parameters
    for param in resolver.model.parameters():
        param.requires_grad = False

    decay_params = ["head.classifier.4.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in resolver.named_parameters()
                if any(nd in n for nd in decay_params)
            ],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in resolver.named_parameters()
                if not any(nd in n for nd in decay_params)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon,
    )

    trainer = Trainer(
        model=resolver,
        args=train_args,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        train_dataset=train_set,
        eval_dataset=valid_set,
    )

    trainer.train()

    # experiment.log_model('Coref with BERT', OUTPUT_PATH / 'checkpoints')

    system_output = Path(train_args.logging_dir) / "gap-system-output.tsv"
    with open(system_output, "w") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        outputs = trainer.predict(test_dataset=test_set)
        pred_probs = F.softmax(torch.from_numpy(outputs.predictions), dim=-1)
        pred_labels = pred_probs.max(dim=-1).indices

        for j, label in enumerate(pred_labels):
            a_coref = "True" if label == 1 else "False"
            b_coref = "True" if label == 2 else "False"
            writer.writerow([test_set_ids[j], a_coref, b_coref])

    experiment.log_asset(system_output)


if __name__ == "__main__":
    run()
