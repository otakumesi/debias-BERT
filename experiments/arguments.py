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
    mixout_prob: float = field(
        default=0,
        metadata={"help": "mixout probaility in all linear layers"}
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default=None, metadata={"help": "Path to dataset"})
    dataset_length: Optional[int] = field(
        default=None, metadata={"help": "Number to use data on your task"}
    )
    augment_data: bool = field(default=True, metadata={"help": "Flag of augmenting dataset"})


@dataclass
class SeqClassificationDataArguments:
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    is_debias: bool = field(
        default=False,
        metadata={"help": "Whether to use debiased dataset"}
    )
