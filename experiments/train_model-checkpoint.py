import logging
import random
from datetime import datetime

from datasets import load_dataset
from transformers import HfArgumentparser, AutoModelWithLMHead, AutoConfig, AutoTokenizer
import torch
from typing import List, Tuple, Dict, Iterable

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

    dataset = load_dataset('csv', data_args.dataset_path)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(train_args.seed)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)


if __name__ == '__main__':
    train()
