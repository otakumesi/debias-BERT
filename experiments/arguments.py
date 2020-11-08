from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "dropout probability in hidden encoders"}
    )
    attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "dropout probability in attentions"}
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default=None, metadata={"help": "Path to dataset"})
    dataset_length: Optional[int] = field(
        default=None, metadata={"help": "Number to use data on your task"}
    )


@dataclass
class MyTrainingArguments:
    seed: int = field(default=42, metadata={"help": "Initialized random seed number"})
    train_batch_size: int = field(
        default=32, metadata={"help": "Batch Size for training"}
    )
    test_batch_size: int = field(default=32, metadata={"help": "Batch Size for test"})
    num_epochs: int = field(default=20, metadata={"help": "Epoch Number for traing"})
    learning_rate: float = field(default=10 - 5, metadata={"help": "Learning Rate"})


@dataclass
class WinobiasDataArguments:
    task_type_number: int = field(
        default=1, metadata={"help": "The number of task type in WinoBias"}
    )
