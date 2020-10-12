import logging
import random
from datetime import datetime

import numpy as np
from datasets import load_dataset
from transformers import AutoModelWithLMHead, AutoConfig, AutoTokenizer
from transformers import HfArgumentParser, Trainer
import torch
from torch.optim import SGD

from arguments import ModelArguments, DataArguments, TrainingArguments
from models import DebiasLoss

ARGS_JSON_FILE = 'args.json'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(level=logging.INFO)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_json_file(ARGS_JSON_FILE)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    dataset = load_dataset('json', data_args.dataset_path)

    # Load Transformers config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # Load Transformers Tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenzier_name, cache_dir=model_args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir)
    loss = DebiasLoss(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(train_args.seed)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dataset.map(lambda example: {f'biased_{k}':v for k, v in tokenizer(example['biased_sentence'])})
    dataset.map(lambda example: {f'base_{k}':v for k, v in tokenizer(example['standard_sentence'])})

    dataset.map(lambda example: {'first_ids': tokenizer.convert_tokens_to_ids(example['targets'][0])})
    dataset.map(lambda example: {'second_ids': tokenizer.convert_tokens_to_ids(example['targets'][1])})

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        optimizers=(SGD([optimizer_grouped_parameters], lr=train_args.learning_rate)),
    )

    trainer.train()
    trainer.save_model(train_args.output_dir)


if __name__ == '__main__':
    train()
