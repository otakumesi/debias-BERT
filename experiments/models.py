from typing import List, Dict, Any, Optional

import torch
from torch import Tensor, IntTensor, LongTensor, BoolTensor
import torch.nn.functional as F
from torch.nn import Module, Sequential, LSTM, ReLU, Linear, Dropout, BatchNorm1d, LayerNorm
from overrides import overrides

from allennlp_models.coref.models.coref import CoreferenceResolver
from allennlp.models.model import Model
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder


class AttentionDebiaser(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self,
                orig_input_ids: Tensor,
                orig_token_type_ids: Tensor,
                orig_attention_mask: Tensor,
                swapped_input_ids: Tensor,
                swapped_token_type_ids: Tensor,
                swapped_attention_mask: Tensor
                ):
        orig_outputs = self.model(input_ids=orig_input_ids,
                                  token_type_ids=orig_token_type_ids,
                                  attention_mask=orig_attention_mask,
                                  output_attentions=True,
                                  return_dict=True)

        swapped_outputs = self.model(input_ids=swapped_input_ids,
                                     token_type_ids=swapped_token_type_ids,
                                     attention_mask=swapped_attention_mask,
                                     output_attentions=True,
                                     return_dict=True)

        orig_attns = torch.stack(
            orig_outputs.attentions).permute(1, 0, 2, 3, 4)
        swapped_attns = torch.stack(
            swapped_outputs.attentions).permute(1, 0, 2, 3, 4)

        # Jensen–Shannon divergence
        js_divs = 1/2 * F.kl_div(orig_attns, swapped_attns, reduction='none') + \
            1/2 * F.kl_div(swapped_attns, orig_attns, reduction='none')
        return torch.sum(js_divs)


class BiasLogProbabilityDebiaser(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self,
                mask_indeces: Tensor,
                first_ids: Tensor,
                second_ids: Tensor,
                biased_input_ids: Tensor,
                biased_token_type_ids: Tensor,
                biased_attention_mask: Tensor,
                base_input_ids: Tensor,
                base_token_type_ids: Tensor,
                base_attention_mask: Tensor
                ):

        biased_log_probs = self.calc_log_probs(
            biased_input_ids, biased_token_type_ids, biased_attention_mask)
        base_log_probs = self.calc_log_probs(
            base_input_ids, base_token_type_ids, base_attention_mask)

        mask_indeces = mask_indeces.view(-1, 1, 1)
        biased_mask_log_probs = biased_log_probs.gather(
            index=mask_indeces.repeat(1, 1, biased_log_probs.shape[2]), dim=1)
        base_mask_log_probs = base_log_probs.gather(
            index=mask_indeces.repeat(1, 1, base_log_probs.shape[2]), dim=1)

        first_increased_log_probs = self.calc_increased_log_prob_scores(biased_mask_log_probs,
                                                                        base_mask_log_probs,
                                                                        first_ids)
        second_increased_log_probs = self.calc_increased_log_prob_scores(biased_mask_log_probs,
                                                                         base_mask_log_probs,
                                                                         second_ids)

        return (F.mse_loss(first_increased_log_probs, second_increased_log_probs),)


class MLPHead(Module):
    def __init__(self, n_input: int, n_targets: int, n_classes: int, dropout=0.2, n_hidden=512):
        super().__init__()

        self.classifier = Sequential(
            Linear(n_input, n_hidden),
            LayerNorm(n_hidden),
            BatchNorm1d(n_targets),
            ReLU(),
            Dropout(dropout),
            Linear(n_hidden, n_classes)
        )

    def forward(self, spans):
        return self.classifier(spans)


class MyCorefResolver(Module):
    # The code is inspired The Model inspired Tenney et al. (2019)
    # https://openreview.net/forum?id=SJzSgnRcKX
    # https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline
    # https://www.kaggle.com/ceshine/pytorch-bert-baseline-public-score-0-54

    def __init__(self, model: Module, n_classes: int = 2, n_hidden: int = 1024):
        super().__init__()

        self.model = model
        self.config = model.config

        # spanをただ単にconcatする方法もありな気がした
        # 例のkaggleを参考に
        self.span_extractor_1 = SelfAttentiveSpanExtractor(
            input_dim=model.config.hidden_size)
        self.span_extractor_2 = SelfAttentiveSpanExtractor(
            input_dim=model.config.hidden_size)

        d_span_1 = self.span_extractor_1.get_output_dim()
        d_span_2 = self.span_extractor_2.get_output_dim()
        self.head = MLPHead(d_span_1 + d_span_2, 2, n_classes)

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids: Tensor,
                p_span_indeces: LongTensor,
                a_span_indeces: LongTensor,
                b_span_indeces: LongTensor,
                labels: Optional[BoolTensor]
                ):

        p_indeces = p_span_indeces.unsqueeze(1)
        a_indeces = a_span_indeces.unsqueeze(1)
        b_indeces = b_span_indeces.unsqueeze(1)

        a_span_pair_indeces = torch.cat((a_indeces, p_indeces), dim=1)
        b_span_pair_indeces = torch.cat((b_indeces, p_indeces), dim=1)

        model_outputs, _ = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        spans_1 = self.span_extractor_1(model_outputs, a_span_pair_indeces)
        spans_2 = self.span_extractor_2(model_outputs, b_span_pair_indeces)

        concatted_spans = torch.cat((spans_1, spans_2), -1)
        logits = self.head(concatted_spans)
        loss = F.cross_entropy(logits, labels.long())

        return loss, logits


class AllenNLPCorefResolver(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_name: str = None,
                 override_weights_file: str = None):
        super().__init__(vocab)

        token_embedder = PretrainedTransformerEmbedder(
            model_name=model_name, override_weights_file=override_weights_file)
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders={'tokens': token_embedder})

        # Hyperparameters is referred to the original paper and BERT analysis paper.
        # https://www.aclweb.org/anthology/D17-1018.pdf
        # https://arxiv.org/pdf/1908.09091.pdf
        feature_size = 20
        embedding_dim = token_embedder.get_output_dim()
        context_layer = PytorchSeq2SeqWrapper(LSTM(input_size=embedding_dim,
                                                   hidden_size=200,
                                                   num_layers=1,
                                                   dropout=0.3,
                                                   bidirectional=True,
                                                   batch_first=True))
        mention_feedforward = FeedForward(input_dim=embedding_dim + 2 * context_layer.get_output_dim() + feature_size,
                                          num_layers=2,
                                          hidden_dims=[150, 150],
                                          activations=[ReLU(), ReLU()],
                                          dropout=[0.3, 0.3])
        antecedent_feedforward = FeedForward(input_dim=mention_feedforward.get_input_dim() * 3 + feature_size,
                                             num_layers=2,
                                             hidden_dims=[150, 150],
                                             activations=[ReLU(), ReLU()],
                                             dropout=[0.3, 0.3])

        self.resolver = CoreferenceResolver(vocab=vocab,
                                            text_field_embedder=text_field_embedder,
                                            context_layer=context_layer,
                                            mention_feedforward=mention_feedforward,
                                            antecedent_feedforward=antecedent_feedforward,
                                            feature_size=feature_size,
                                            max_span_width=10,
                                            spans_per_word=0.4,
                                            max_antecedents=250)

    @overrides
    def forward(self,
                text: TextFieldTensors,
                spans: IntTensor,
                metadata: List[Dict[str, Any]],
                span_labels: IntTensor = None
                ) -> Dict[str, Tensor]:
        return self.resolver(text=text,
                             spans=spans,
                             span_labels=span_labels,
                             metadata=metadata)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.resolver.get_metrics(reset)
