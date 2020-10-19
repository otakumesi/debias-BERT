from typing import Tuple, Any
from torch import Tensor
import torch.nn.functional as F
from torch.nn import LSTM, ReLU

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl

from allennlp_models import CoreferenceResolver
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper


class DebiasModule(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizers: Tuple[Optimizer, _LRScheduler]):
        super().__init__() self.model = model
        self.optimizers = optimizers

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

        biased_log_probs = self.calc_log_probs(biased_input_ids, biased_token_type_ids, biased_attention_mask)
        base_log_probs = self.calc_log_probs(base_input_ids, base_token_type_ids, base_attention_mask)

        mask_indeces = mask_indeces.view(-1, 1, 1)
        biased_mask_log_probs = biased_log_probs.gather(index=mask_indeces.repeat(1, 1, biased_log_probs.shape[2]), dim=1)
        base_mask_log_probs = base_log_probs.gather(index=mask_indeces.repeat(1, 1, base_log_probs.shape[2]), dim=1)

        first_increased_log_probs = self.calc_increased_log_prob_scores(biased_mask_log_probs,
                                                                        base_mask_log_probs,
                                                                        first_ids)
        second_increased_log_probs = self.calc_increased_log_prob_scores(biased_mask_log_probs,
                                                                         base_mask_log_probs,
                                                                         second_ids)

        return (F.mse_loss(first_increased_log_probs, second_increased_log_probs),)

    def calc_log_probs(self, input_ids, token_type_ids, attention_mask):
        inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        outputs = self.model(**inputs, return_dict = True)
        logits = outputs.logits
        return F.log_softmax(logits)


    def calc_increased_log_prob_scores(self,
                                       biased_mask_log_probs,
                                       base_mask_log_probs,
                                       target_ids):
        target_ids = target_ids.view(-1, 1, 1)
        p_tgt = biased_mask_log_probs.gather(index=target_ids, dim=2)
        p_prior = base_mask_log_probs.gather(index=target_ids, dim=2)
        return p_tgt - p_prior

    def training_step(self, batch, batch_idx):
        loss = self.forward(**batch)
        return {'loss': loss }

    def configure_optimizers(self):
        optimizer, scheduler = self.optimizers
        return [optimizer], [scheduler]


class CorefModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        # Hyperparameters is referred to the original paper and BERT analysis paper.
        # https://www.aclweb.org/anthology/D17-1018.pdf
        # https://arxiv.org/pdf/1908.09091.pdf
        vocab = Vocaburary()
        feature_size = 20
        context_layer = PytorchSeq2SeqWrapper(LSTM(nput_size=embedding_dim),
                                              hidden_size=200,
                                              num_layers=1,
                                              dropout=0.3,
                                              bidirectional=True,
                                              batch_first=True)
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
                                            text_field=text_field,
                                            context_layer=context_layer,
                                            mention_feedforward=mention_feedforward,
                                            antecedent_feedforward=antecedent_feedforward,
                                            feature_size=feature_size,
                                            max_span_width=10,
                                            spans_per_word=0.4,
                                            max_antecedent=250)
