import logging
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from transformers import HfArgumentParser
import torch
from torch.optim import Adam
from allennlp_models.coref.dataset_readers.conll import ConllCorefReader
from allennlp_models.coref.metrics.conll_coref_scores import ConllCorefScores
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data import Vocabulary
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.util import evaluate
from dotenv import load_dotenv

from arguments import ModelArguments, WinobiasDataArguments, TrainingArguments
from models import MyCorefResolver

ARGS_JSON_FILE = "args.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(level=logging.INFO)


DATASET_PATH = Path("data/winobias")
OUTPUT_PATH = PATH('runs') / TIMESTAMP


def run():
    parser = HfArgumentParser(
        (ModelArguments, WinobiasDataArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_json_file(ARGS_JSON_FILE)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    tokenizer = PretrainedTransformerTokenizer(
        model_name=model_args.model_name_or_path)
    token_indexer = PretrainedTransformerIndexer(
        model_name=model_args.model_name_or_path
    )

    dataset_reader = ConllCorefReader(
        max_span_width=10,
        wordpiece_modeling_tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
    )

    t_num = data_args.task_type_number
    file_name_template = "{0}_type{1}_{2}_stereotype.v4_auto_conll"

    dataset_attrs = ["anti", "pro"]

    for ds_attr in dataset_attrs:
        train_set = dataset_reader.read(
            DATASET_PATH / file_name_template.format("dev", t_num, ds_attr)
        )
        test_set = dataset_reader.read(
            DATASET_PATH / file_name_template.format("test", t_num, ds_attr)
        )

        vocab = Vocabulary()
        train_set.index_with(vocab)
        test_set.index_with(vocab)

        train_loader = PyTorchDataLoader(
            train_set, batch_size=train_args.train_batch_size
        )
        test_loader = PyTorchDataLoader(
            train_set, batch_size=train_args.test_batch_size
        )

        model = MyCorefResolver(
            vocab=vocab,
            model_name=model_args.model_name_or_path,
            override_weights_file=model_args.override_weights_file,
        )

        optimizer = Adam(model.parameters(), lr=train_args.learning_rate)

        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            num_epochs=train_args.num_epochs,
            cuda_device=1 if torch.cuda.is_available() else -1
        )

        trainer.train()

        try:
            OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_PATH / f'{ds_attr}_stereotype_result.txt'
            evaluate(model=model,
                     data_loader=test_loader,
                     output_file=output_file,
                     cuda_device=1 if torch.cuda.is_available() else -1)
        except BaseException as e:
            OUTPUT_PATH.rmdir()
            raise e


if __name__ == "__main__":
    run()
