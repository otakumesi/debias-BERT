from comet_ml import Experiment

import logging
import random
from pathlib import Path
import difflib
import re

import numpy as np
from datasets import load_dataset
from transformers import AutoModelWithLMHead, AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers import TrainingArguments, Trainer, AdamW
import torch
from torch import nn
from dotenv import load_dotenv

from arguments import ModelArguments, DataArguments
from models import SentencePertubationNormalizer
from mixout import MixLinear

ARGS_JSON_FILE = "args_train_model.json"
logging.basicConfig(level=logging.INFO)
load_dotenv()

DATASET_PATH = Path("data") / "crows_pairs_train.csv"


def train(model_args, data_args, train_args):
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # Load Transformers config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob
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
    model = SentencePertubationNormalizer(model)

    # Integrate Mixout
    if model_args.mixout_prob > 0:
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict["weight"], model_args.mixout_prob
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)

    MAX_LEN = 70
    dataset = load_dataset("csv", data_files=str(DATASET_PATH), split="train")

    def augment_dataset(exs):
        sents_more = []
        sents_less = []
        stereo_antistereo_list = []
        bias_types = []
        for sent_more, sent_less, stereo_antistereo, bias_type in zip(exs["sent_more"], exs["sent_less"], exs["stereo_antistereo"], exs["bias_type"]):
            sents_more.extend([sent_more, sent_less])
            sents_less.extend([sent_less, sent_more])
            stereo_antistereo_list.extend([stereo_antistereo] * 2)
            bias_types.extend([bias_type] * 2)

        return {
            "sent_more": sents_more,
            "sent_less": sents_less,
            "stereo_antistereo": stereo_antistereo_list,
            "bias_type": bias_types
        }

    if data_args.augment_data == True:
        dataset = dataset.map(augment_dataset, remove_columns=['Unnamed: 0', 'annotations', 'anon_writer', 'anon_annotators'], batched=True)

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
        max_len = MAX_LEN

        m_input_ids = [str(x) for x in ex["more_input_ids"][1:]]
        l_input_ids = [str(x) for x in ex["less_input_ids"][1:]]

        matcher = difflib.SequenceMatcher(None, m_input_ids, l_input_ids)
        more_result, less_result = [], []

        for op, m_start_pos, m_end_pos, l_start_pos, l_end_pos in matcher.get_opcodes():
            if op == 'equal':
                more_result += [x for x in range(m_start_pos, m_end_pos, 1)]
                less_result += [x for x in range(l_start_pos, l_end_pos, 1)]

        more_mask = torch.ones(max_len, dtype=torch.int64)
        more_len = len(more_result)
        if more_len < max_len:
            more_diff = max_len - more_len
            more_mask.scatter_(0, torch.LongTensor(range(more_len, more_len + more_diff)), 0)
            more_result.extend([max_len - 1] * more_diff)

        less_mask = torch.ones(max_len, dtype=torch.int64)
        less_len = len(less_result)
        if less_len < max_len:
            less_diff = max_len - less_len
            less_mask.scatter_(0, torch.LongTensor(range(less_len, less_len + less_diff)), 0)
            less_result.extend([max_len - 1] * less_diff)

        return {'more_indices': more_result,
                'more_mask': more_mask,
                'less_indices': less_result,
                'less_mask': less_mask}

    dataset = dataset.map(get_indices)

    orig_columns = ["input_ids", "token_type_ids", "attention_mask", "indices", "mask"]
    columns = [f"more_{c}" for c in orig_columns] + [f"less_{c}" for c in orig_columns]
    dataset.set_format(type="torch", columns=columns)

    # model_original_parameters = {k:v for k, v in model.state_dict().items() if re.search(r"\w+\.layer\.\d+\..+\.(weight)", k)}
    # {
    #     "params": [
    #         p - model_original_parameters[n]
    #         for n, p in model.named_parameters()
    #         if n in model_original_parameters
    #     ],
    #     "weight_decay": model_args.regular_weight_decay
    # }

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
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        optimizers=(optimizer, None),
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
