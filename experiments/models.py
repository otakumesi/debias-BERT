import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.linalg as LA


class SentencePertubationNormalizer(nn.Module):
    def __init__(self, model: nn.Module, k: float = 0.01):
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

        more_hidden_state = more_outputs.hidden_states[-1]

        less_hidden_state = less_outputs.hidden_states[-1]

        loss = (more_hidden_state.softmax(dim=-1) * F.mse_loss(less_hidden_state, more_hidden_state.detach(), reduction='none')).sum(dim=-1).mean()

        return (loss,)


class BertEmbeddingsWithDebias(nn.Module):
    def __init__(self, config, bias_subspace):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.vocab_size = config.vocab_size

        self.bias_subspace = bias_subspace
        self.bias_subspace.requires_grad = False

        vocab_tensors = self.word_embeddings(torch.tensor(range(0, config.vocab_size)))
        self.normalize_norm = LA.norm(vocab_tensors, dim=0)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.bias_subspace is not None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            subspace_tensor = self.bias_subspace.view(1, 1, -1).repeat_interleave(input_ids.shape[1], dim=1).repeat_interleave(input_ids.shape[0], dim=0).to(device)
            inputs_embeds -= subspace_tensor
            inputs_embeds -= self.normalize_norm.view(1, 1, -1).repeat_interleave(input_ids.shape[1], dim=1).repeat_interleave(input_ids.shape[0], dim=0).to(device)
            inputs_embeds /= LA.norm(inputs_embeds, dim=-1).view(*inputs_embeds.shape[:2], 1).repeat_interleave(inputs_embeds.shape[-1], dim=-1)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
