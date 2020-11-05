from comet_ml import Experiment

import os
import logging
import random
from pathlib import Path
import difflib

import numpy as np
from datasets import load_dataset
from transformers import AutoModelWithLMHead, AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers import TrainingArguments, Trainer, get_constant_schedule
import torch
from torch.optim import SGD
from dotenv import load_dotenv

from arguments import ModelArguments, DataArguments
from models import BiasDivergenceDebiaser

ARGS_JSON_FILE = "args.json"
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("COMET_API_KEY")
WORKSPACE = os.getenv("COMET_WORKSPACE")
PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")

# experiment = Experiment(api_key=API_KEY, workspace=WORKSPACE, project_name=PROJECT_NAME)

DATASET_PATH = Path("data") / "crows_pairs_anonymized.csv"


def train(model_args, data_args, train_args):
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    dataset = load_dataset("csv", data_files=str(DATASET_PATH), split="train")

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
            model_args.tokenzier_name, cache_dir=model_args.cache_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir
    )
    debiaser = BiasDivergenceDebiaser(model)

    MAX_LEN = 100
    dataset = dataset.map(
        lambda ex: {
            f"more_{k}": v
            for k, v in tokenizer(
                ex["sent_more"], max_length=MAX_LEN, padding="max_length"
            ).items()
        }
    )
    dataset = dataset.map(
        lambda ex: {
            f"less_{k}": v
            for k, v in tokenizer(
                ex["sent_less"], max_length=MAX_LEN, padding="max_length"
            ).items()
        }
    )

    def get_indices(ex):
        m_input_ids = [str(x) for x in ex["more_input_ids"]]
        l_input_ids = [str(x) for x in ex["less_input_ids"]]

        matcher = difflib.SequenceMatcher(None, m_input_ids, l_input_ids)
        more_result, less_result = [], []

        for op, m_start_pos, m_end_pos, l_start_pos, l_end_pos in matcher.get_opcodes():
            if op == 'equal':
                more_result += [x for x in range(m_start_pos, m_end_pos, 1)]
                less_result += [x for x in range(l_start_pos, l_end_pos, 1)]

        more_mask = torch.ones(MAX_LEN, dtype=torch.int64)
        more_len = len(more_result)
        if more_len < MAX_LEN:
            more_diff = MAX_LEN - more_len
            more_mask.scatter_(0, torch.LongTensor(range(more_len, more_len + more_diff)), 0)
            more_result.extend([MAX_LEN - 1] * more_diff)
            
        less_mask = torch.ones(MAX_LEN, dtype=torch.int64)
        less_len = len(less_result)
        if less_len < MAX_LEN:
            less_diff = MAX_LEN - less_len
            less_mask.scatter_(0, torch.LongTensor(range(less_len, less_len + less_diff)), 0)
            less_result.extend([MAX_LEN - 1] * less_diff)

        return {'more_indices': more_result,
                'more_mask': more_mask,
                'less_indices': less_result,
                'less_mask': less_mask}

    dataset = dataset.map(get_indices)

    # dataset = dataset.filter(lambda ex: len([i for i in ex["more_input_ids"] if i != 0]) == len([i for i in  ex["less_input_ids"] if i != 0]))

    # dataset = dataset.map(
    #     lambda ex: {"unmodified_mask": [m == l for m, l in zip(ex["more_input_ids"], ex["less_input_ids"])]}
    # )

    orig_columns = ["input_ids", "token_type_ids", "attention_mask", "indices", "mask"]
    columns = [f"more_{c}" for c in orig_columns] + [f"less_{c}" for c in orig_columns] # + ["unmodified_mask"]
    dataset.set_format(type="torch", columns=columns)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = SGD(optimizer_grouped_parameters, lr=train_args.learning_rate)
    lr_scheduler = get_constant_schedule(optimizer)

    trainer = Trainer(
        model=debiaser,
        args=train_args,
        train_dataset=dataset,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()

    model_output_dir = Path(train_args.logging_dir)
    trainer.model.model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_json_file(ARGS_JSON_FILE)

    if train_args.do_train:
        train(model_args, data_args, train_args)
