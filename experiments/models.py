from typing import List, Tuple, Dict, Iterable
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class DebiasLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super(DebiasLoss, self).__init__()
        self.model = model

    def forward(self,
                inputs: Iterable[Dict[str, Tensor]],
                first_ids: List[Tensor],
                second_ids: List[Tensor]):
        mask_indeces = inputs.pop('mask_idx')
        outputs = self.model(**inputs, return_dict = True)
        logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=2)

        mask_indeces = mask_indeces.view(-1, 1)
        mask_log_probs = log_probs.gather(index=mask_indeces, dim=1)

        first_ids = first_ids.view(-1, 1)
        first_tkn_log_probs = mask_log_probs.gather(index=first_ids, dim=2)

        second_ids = second_ids.view(-1, 1)
        second_tkn_log_probs = mask_log_probs.gather(index=second_ids, dim=2)

        losses = F.relu(first_tkn_log_probs - second_tkn_log_probs)
        return losses.mean()
