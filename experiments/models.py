from typing import List, Tuple, Dict, Iterable
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from utils import extract_kv_by_prefix

class DebiasLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super(DebiasLoss, self).__init__()
        self.model = model

    def forward(self,
                inputs: Iterable[Dict[str, Tensor]],
                first_ids: Tensor,
                second_ids: Tensor,
                biased_prefix: str = 'biased_',
                base_prefix: str = 'base_'
    ):

        mask_indeces = inputs.pop('mask_idx')

        biased_log_probs = self.calc_log_probs(inputs, biased_prefix)
        base_log_probs = self.calc_log_probs(inputs, base_prefix)

        mask_indeces = mask_indeces.view(-1, 1)
        biased_mask_log_probs = biased_log_probs.gather(index=mask_indeces, dim=1)
        base_mask_log_probs = base_log_probs.gather(index=mask_indeces, dim=1)

        first_increased_log_probs = self.calc_increased_log_prob_scores(biased_mask_log_probs,
                                                                        base_mask_log_probs,
                                                                        first_ids)
        second_increased_log_probs = self.calc_increased_log_prob_scores(biased_mask_log_probs,
                                                                         base_mask_log_probs,
                                                                         second_ids)

        return F.mse_loss(first_increased_log_probs - second_increased_log_probs)

    def calc_log_probs(self, inputs, prefix):
        target_inputs = extract_kv_by_prefix(inputs, prefix)
        outputs = self.model(**target_inputs, return_dict = True)
        logits = outputs.logits
        return F.log_softmax(logits)


    def calc_increased_log_prob_scores(self,
                                       biased_mask_log_probs,
                                       base_mask_log_probs,
                                       target_ids):
        target_ids.view(-1, 1)
        p_tgt = biased_mask_log_probs.gather(index=target_ids, dim=2)
        p_prior = base_mask_log_probs.gather(index=target_ids, dim=2)
        return p_tgt - p_prior
