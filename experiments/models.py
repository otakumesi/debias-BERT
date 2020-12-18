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

        more_logits = more_outputs.logits * more_attention_mask.unsqueeze(2)
        less_logits = less_outputs.logits * less_attention_mask.unsqueeze(2)

        vocab_size = self.model.config.vocab_size

        # batch_size = more_input_ids.shape[0]
        # seq_size = more_attention_mask.shape[1]
        # more_attentions = torch.stack(more_outputs.attentions).permute(1, 0, 2, 3, 4)
        # more_attentions *= more_attention_mask.view(batch_size, 1, 1, 1, seq_size)

        # less_attentions = torch.stack(less_outputs.attentions).permute(1, 0, 2, 3, 4)
        # less_attentions *= less_attention_mask.view(batch_size, 1, 1, 1, seq_size)

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

        more_probs = torch.softmax(more_logits, dim=-1).detach()
        less_probs = torch.softmax(less_logits, dim=-1)

        # more_log_probs = torch.log(more_probs)
        # less_log_probs = torch.log(less_probs)

        # more_log_logits = torch.log(more_logits + 1)
        # less_log_logits = torch.log(less_logits + 1)

        # loss = F.kl_div(more_log_probs, less_log_probs.detach(), log_target=True, reduction="batchmean")

        M = (more_probs + less_probs) * 0.5
        left_kl = F.kl_div(more_probs.log(), M, reduction="none") * 0.5
        right_kl = F.kl_div(less_probs.log(), M, reduction="none") * 0.5
        loss = (left_kl + right_kl).sum(dim=[-1, -2]).mean()


        return (loss,)
