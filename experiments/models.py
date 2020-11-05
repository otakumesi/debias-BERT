from typing import Optional

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn.functional as F
from torch.nn import Module, Sequential, ReLU, Linear, Dropout, BatchNorm1d
from allennlp.nn.util import batched_span_select


class CosineDebiaser(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        more_input_ids: Tensor,
        more_token_type_ids: Tensor,
        more_attention_mask: Tensor,
        less_input_ids: Tensor,
        less_token_type_ids: Tensor,
        less_attention_mask: Tensor,
        unmodified_mask: Tensor
    ):
        more_outputs = self.model(
            input_ids=more_input_ids,
            token_type_ids=more_token_type_ids,
            attention_mask=more_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        less_outputs = self.model(
            input_ids=less_input_ids,
            token_type_ids=less_token_type_ids,
            attention_mask=less_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_layer = self.config.num_hidden_layers
        more_hiddens = more_outputs.hidden_states[last_layer]
        less_hiddens = less_outputs.hidden_states[last_layer]

        # reject modified tokens
        unmodified_mask = unmodified_mask.unsqueeze(2).repeat_interleave(dim=2, repeats=self.config.hidden_size)
        more_hiddens *= unmodified_mask
        less_hiddens *= unmodified_mask

        loss = torch.mean(torch.sum(1 / 2 * ((1 - F.cosine_similarity(more_hiddens, less_hiddens, dim=2)) ** 2), dim=1))
        return (loss,)


class AttentionDebiaser(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        more_input_ids: Tensor,
        more_token_type_ids: Tensor,
        more_attention_mask: Tensor,
        less_input_ids: Tensor,
        less_token_type_ids: Tensor,
        less_attention_mask: Tensor,
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

        more_attns = torch.stack(more_outputs.attentions).permute(1, 0, 2, 3, 4)
        less_attns = torch.stack(less_outputs.attentions).permute(1, 0, 2, 3, 4)
        attn_size = self.config.num_attention_heads * self.config.num_hidden_layers

        # Jensen–Shannon divergence
        # js_divs = 1 / 2 * F.kl_div(
        #     more_attns, less_attns, reduction="none"
        # ) + 1 / 2 * F.kl_div(less_attns, more_attns, reduction="none")
        # return (torch.sum(js_divs),)

        return (F.kl_div(more_attns, less_attns, reduction="batchmean") / attn_size,)


class BiasLogProbabilityDebiaser(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        mask_indeces: Tensor,
        first_ids: Tensor,
        second_ids: Tensor,
        biased_input_ids: Tensor,
        biased_token_type_ids: Tensor,
        biased_attention_mask: Tensor,
        base_input_ids: Tensor,
        base_token_type_ids: Tensor,
        base_attention_mask: Tensor,
    ):

        biased_log_probs = self.calc_log_probs(
            biased_input_ids, biased_token_type_ids, biased_attention_mask
        )
        base_log_probs = self.calc_log_probs(
            base_input_ids, base_token_type_ids, base_attention_mask
        )

        mask_indeces = mask_indeces.view(-1, 1, 1)
        biased_mask_log_probs = biased_log_probs.gather(
            index=mask_indeces.repeat(1, 1, biased_log_probs.shape[2]), dim=1
        )
        base_mask_log_probs = base_log_probs.gather(
            index=mask_indeces.repeat(1, 1, base_log_probs.shape[2]), dim=1
        )

        first_increased_log_probs = self.calc_increased_log_prob_scores(
            biased_mask_log_probs, base_mask_log_probs, first_ids
        )
        second_increased_log_probs = self.calc_increased_log_prob_scores(
            biased_mask_log_probs, base_mask_log_probs, second_ids
        )

        return (F.mse_loss(first_increased_log_probs, second_increased_log_probs),)


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
