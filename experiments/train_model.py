from comet_ml import Experiment

import os
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoModelWithLMHead, AutoConfig, AutoTokenizer
from transformers import HfArgumentParser
from transformers import TrainingArguments
import torch
from torch.optim import SGD
from dotenv import load_dotenv

from arguments import ModelArguments, DataArguments
from models import AttentionDebiaser
from trainers import MyFineTuneTrainer

ARGS_JSON_FILE = 'args.json'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(level=logging.INFO)

load_dotenv()

API_KEY = os.getenv('COMET_API_KEY')
WORKSPACE = os.getenv('COMET_WORKSPACE')
PROJECT_NAME = os.getenv('COMET_PROJECT_NAME')

experiment = Experiment(api_key=API_KEY,
                        workspace=WORKSPACE,
                        project_name=PROJECT_NAME,
                        auto_output_logging='simple')

OUTPUT_PATH = Path('runs') / TIMESTAMP


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_json_file(ARGS_JSON_FILE)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    dataset = load_dataset(
        'json', data_files=data_args.dataset_path, field='data', split='train')

    # Load Transformers config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir)
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # Load Transformers Tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenzier_name, cache_dir=model_args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir)
    loss = AttentionDebiaser(model)

    device = 'cuda' if torch.cuda.is_avaiable() else 'cpu'
    loss.to(device)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(train_args.seed)
    if n_gpu > 1:
        loss = torch.nn.DataParallel(loss)

    MAX_LEN = 50
    dataset = dataset.map(lambda example: {f'biased_{k}': v for k, v in tokenizer(
        example['biased_sentence'], max_length=MAX_LEN, padding='max_length').items()})
    dataset = dataset.map(lambda example: {f'base_{k}': v for k, v in tokenizer(
        example['base_sentence'], max_length=MAX_LEN, padding='max_length').items()})
    dataset = dataset.map(lambda example: {
                          'mask_indeces': example['biased_input_ids'].index(tokenizer.mask_token_id)})

    dataset = dataset.map(lambda example: {'first_ids': tokenizer.convert_tokens_to_ids(example['targets'][0]),
                                           'second_ids': tokenizer.convert_tokens_to_ids(example['targets'][1])})

    simple_columns = ['input_ids', 'token_type_ids', 'attention_mask']
    columns = [f'biased_{c}' for c in simple_columns] + [
        f'base_{c}' for c in simple_columns] + ['first_ids', 'second_ids', 'mask_indeces']
    dataset.set_format(type='torch', columns=columns)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in loss.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_args.weight_decay
        },
        {
            "params": [p for n, p in loss.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    trainer = MyFineTuneTrainer(
        model=loss,
        args=train_args,
        train_dataset=dataset,
        optimizers=(SGD(optimizer_grouped_parameters,
                        lr=train_args.learning_rate), None),
    )

    trainer.train()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_PATH)
    experiment.log_model('DebiasBERT', OUTPUT_PATH)


if __name__ == '__main__':
    train()
