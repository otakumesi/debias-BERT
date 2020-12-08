from typing import Optional

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn.functional as F
from torch.nn import Module, Sequential, ReLU, Linear, Dropout, BatchNorm1d
from allennlp.nn.util import batched_span_select


class SentencePertubationNormalizer(Module):
    def __init__(self, model: Module, k: float = 0.01):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        more_input_ids: Tensor,
        more_token_type_ids: Tensor,
        more_attention_mask: Tensor,
        more_indices: Tensor,
        more_mask: Tensor,
        less_input_ids: Tensor,
        less_token_type_ids: Tensor,
        less_attention_mask: Tensor,
        less_indices: Tensor,
        less_mask: Tensor
    ):
        more_outputs = self.model(
            input_ids=more_input_ids,
            token_type_ids=more_token_type_ids,
            attention_mask=more_attention_mask,
            output_attentions=True,
            return_dict=True,
        )

        less_outputs = self.model(
            input_ids=less_input_ids,
            token_type_ids=less_token_type_ids,
            attention_mask=less_attention_mask,
            output_attentions=True,
            return_dict=True,
        )

        # more_logits = more_outputs.logits * more_attention_mask.unsqueeze(2)
        # less_logits = less_outputs.logits * less_attention_mask.unsqueeze(2)

        vocab_size = self.model.config.vocab_size

        # batch_size = more_input_ids.shape[0]
        # seq_size = more_attention_mask.shape[1]
        # more_attentions = torch.stack(more_outputs.attentions).permute(1, 0, 2, 3, 4)
        # more_attentions *= more_attention_mask.view(batch_size, 1, 1, 1, seq_size)

        # less_attentions = torch.stack(less_outputs.attentions).permute(1, 0, 2, 3, 4)
        # less_attentions*= less_attention_mask.view(batch_size, 1, 1, 1, seq_size)

        # more_attention_indeces = more_indices.view(batch_size, 1, 1, 1, seq_size)
        # less_attention_indeces = less_indices.view(batch_size, 1, 1, 1, seq_size)
        # more_attentions = more_attentions.gather(dim=4, index=more_attention_indeces).view(batch_size, -1)
        # less_attentions = less_attentions.gather(dim=4, index=less_attention_indeces).view(batch_size, -1)

        # more_attentions = more_attentions.where(more_attentions <= 0, more_attentions.log())
        # less_attentions = less_attentions.where(less_attentions <= 0, less_attentions.log())
        
        more_logits_indices = more_indices.unsqueeze(2).repeat_interleave(dim=2, repeats=vocab_size)
        more_logits = more_logits.gather(dim=1, index=more_logits_indices)
        more_logits = more_logits * more_mask.unsqueeze(2)

        less_logits_indices = less_indices.unsqueeze(2).repeat_interleave(dim=2, repeats=vocab_size)
        less_logits = less_logits.gather(dim=1, index=less_logits_indices)
        less_logits = less_logits * less_mask.unsqueeze(2)

        more_probs = torch.log_softmax(more_logits, dim=-1).detach()
        less_probs = torch.log_softmax(less_logits, dim=-1)

        return (F.mse_loss(more_log_probs, less_log_probs),)


class MLPHead(Module):
    def __init__(self, n_input: int, n_classes: int, n_hidden: int, dropout=None):
        super().__init__()

        self.classifier = Sequential(
            Linear(n_input, n_hidden),
            BatchNorm1d(n_hidden),
            ReLU(),
            Dropout(dropout),
            Linear(n_hidden, n_classes),
        )

    def forward(self, spans):
        return self.classifier(spans)


class MyCorefResolver(Module):
    # The code is inspired The Model inspired Tenney et al. (2019)
    # https://openreview.net/forum?id=SJzSgnRcKX
    # https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline
    # https://www.kaggle.com/ceshine/pytorch-bert-baseline-public-score-0-54

    def __init__(self, model: Module, n_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.model = model
        self.config = model.config

        self.head = MLPHead(
            model.config.hidden_size * 3,
            n_classes,
            model.config.hidden_size * 2 + n_classes,
            dropout,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        p_span_indeces: LongTensor,
        a_span_indeces: LongTensor,
        b_span_indeces: LongTensor,
        labels: Optional[BoolTensor],
    ):

        p_indeces = p_span_indeces.unsqueeze(1)
        a_indeces = a_span_indeces.unsqueeze(1)
        b_indeces = b_span_indeces.unsqueeze(1)

        model_outputs, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        span_p_embeddings = self.calculate_meaned_span_embeddings(
            model_outputs, p_indeces
        )
        span_a_embeddings = self.calculate_meaned_span_embeddings(
            model_outputs, a_indeces
        )
        span_b_embeddings = self.calculate_meaned_span_embeddings(
            model_outputs, b_indeces
        )

        concatted_span_embeddings = torch.cat(
            (span_a_embeddings, span_b_embeddings, span_p_embeddings), -1
        ).squeeze(1)
        logits = self.head(concatted_span_embeddings)
        loss = F.cross_entropy(logits, labels.long())

        return loss, logits

    def calculate_meaned_span_embeddings(self, embeddings, span_indeces):
        span_embeddings, span_mask = batched_span_select(
            embeddings.contiguous(), span_indeces
        )
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask
        summed_span_embeddings = span_embeddings.sum(2)
        summed_span_mask = span_mask.sum(2)

        return summed_span_embeddings / torch.clamp_min(summed_span_mask, 1)
