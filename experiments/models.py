from typing import List, Tuple, Dict, Iterable, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from utils import extract_kv_by_prefix

class DebiasLoss(nn.Module):
    def __init__(self, model: nn.Module, config: Any):
        super(DebiasLoss, self).__init__()
        self.model = model
        self.config = config

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
