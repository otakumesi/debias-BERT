import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module


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

        # more_logits = more_outputs.logits * more_attention_mask.unsqueeze(2)
        # less_logits = less_outputs.logits * less_attention_mask.unsqueeze(2)

        # vocab_size = self.model.config.vocab_size

        # more_logits_indices = more_indices.unsqueeze(2).repeat_interleave(dim=2, repeats=vocab_size)
        # more_logits = more_logits.gather(dim=1, index=more_logits_indices)
        # more_logits = more_logits * more_mask.unsqueeze(2)

        # less_logits_indices = less_indices.unsqueeze(2).repeat_interleave(dim=2, repeats=vocab_size)
        # less_logits = less_logits.gather(dim=1, index=less_logits_indices)
        # less_logits = less_logits * less_mask.unsqueeze(2)

        # more_probs = torch.log_softmax(more_logits, dim=-1)
        # less_probs = torch.log_softmax(less_logits, dim=-1)

        # difference weighted less prob between less log probs and more log probs
        # loss = F.kl_div(less_probs, more_probs.detach(), log_target=True, reduction="batchmean") / less_probs.shape[1]

        hidden_size = self.model.config.hidden_size
        more_hs_indices = more_indices.unsqueeze(2).repeat_interleave(dim=2, repeats=hidden_size)
        more_hidden_state = more_outputs.hidden_states[-1]
        more_hidden_state = more_hidden_state.gather(dim=1, index=more_hs_indices)
        more_hidden_state = more_hidden_state * more_mask.unsqueeze(2)

        less_hs_indices = less_indices.unsqueeze(2).repeat_interleave(dim=2, repeats=hidden_size)
        less_hidden_state = less_outputs.hidden_states[-1]
        less_hidden_state = less_hidden_state.gather(dim=1, index=less_hs_indices)
        less_hidden_state = less_hidden_state * less_mask.unsqueeze(2)

        # loss = F.kl_div(less_probs, more_probs.detach(), log_target=True, reduction="batchmean") / less_probs.shape[1]

        loss = (more_hidden_state.softmax(dim=-1) * F.mse_loss(less_hidden_state, more_hidden_state.detach(), reduction='none')).sum(dim=-1).mean()

        return (loss,)
