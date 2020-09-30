from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    evaluation_method: str = field(
        default='mask_pred_score',
        metadata={'help': 'Evaluation method to pass Integrated Gradients. Ex. mask_pred_score, same_target_loss, another_target_loss'}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Where do you want to store the pretrained models'}
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None,
        metadata={'help': 'Path to dataset'}
    )
    dataset_length: Optional[int]= field(
        default=None,
        metadata={'help': 'Number to use data on your task.'}
    )


@dataclass
class TrainingArguments:
    pass
