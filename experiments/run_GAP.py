import csv
import logging
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from transformers import HfArgumentParser
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from dotenv import load_dotenv
from datasets import load_dataset

from arguments import ModelArguments, WinobiasDataArguments, TrainingArguments
from models import MyCorefResolver
from trainers import CorefRunner
from utils import prepare_gap

ARGS_JSON_FILE = "args.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO)

DATASET_PATH = Path("data/winobias")
OUTPUT_PATH = Path('runs') / TIMESTAMP


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
            model_args.config_name, cache_dir=model_args.cache_dir)
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # Load Transformers Tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenzier_name, cache_dir=model_args.cache_dir, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=True)

    datasets = load_dataset('gap')
    datasets = prepare_gap(datasets, tokenizer)

    train_set = datasets['train']
    valid_set = datasets['validation']
    test_set = datasets['test']

    columns = ['input_ids', 'attention_mask', 'token_type_ids', 'a_span_indeces', 'b_span_indeces', 'labels']
    train_set.set_format(type='torch', columns=columns)
    valid_set.set_format(type='torch', columns=columns)
    test_set.set_format(type='torch', columns=columns)

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)
    resolver = MyCorefResolver(model=model)

    # Fixed pretrained model parameters
    for param in resolver.model.parameters():
        param.requires_grad = False

    optimizer = Adam(resolver.parameters(), lr=train_args.learning_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolver.to(device)

    runner = CorefRunner(model=resolver, device=device)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train_loaders = {'train': DataLoader(train_set, batch_size=train_args.train_batch_size),
                     'valid': DataLoader(valid_set, batch_size=train_args.train_batch_size)}
    # runner.train(
    #     criterion=CrossEntropyLoss,
    #     optimizer=optimizer,
    #     loaders=train_loaders,
    #     num_epochs=train_args.num_epochs,
    #     initial_seed=train_args.seed,
    #     logdir=OUTPUT_PATH
    #     verbose=True
    # )

    test_loader = DataLoader(test_set, batch_size=5)
    with open(OUTPUT_PATH / 'gep-system-output.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        headers = ['ID', 'Pronoun', 'A-coref', 'B-coref']
        writer.writerow(headers)
        for i, preds in enumerate(runner.predict_loader(loader=test_loader)):
            pred_probs = F.softmax(preds, -1)
            pred_labels = pred_probs.max(dim=-1).indeces

            for j, label in enumerate(pred_labels):
                a_coref = 'True' if pred_labels == 1 else 'False'
                b_coref = 'True' if pred_labels == 2 else 'False'

                test_row = datasets['test'][i*train_args.test_batch_size+j]
                writer.writerow(
                    [test_row['ID'], test_row['Pronoun'], a_coref, b_coref])


if __name__ == "__main__":
    run()
