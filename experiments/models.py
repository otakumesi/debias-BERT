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
    def __init__(self, config, scaling_token_ids_set_list, bias_subspace_tensors, k=1):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.vocab_size = config.vocab_size

        self.bias_subspace_tensors = bias_subspace_tensors
        self.bias_subspace_tensors.requires_grad = False

        self.bias_transformations = torch.zeros(config.vocab_size, config.hidden_size)
        self.scaling_coefs = torch.ones(config.vocab_size)
        for i, scaling_token_ids_set in enumerate(scaling_token_ids_set_list):
            bias_subspaces = bias_subspace_tensors[i].float()
            for scaling_ids in scaling_token_ids_set:
                embeddings = self.word_embeddings(torch.tensor(scaling_ids))

                bias_norms = LA.norm(bias_subspaces[:k], dim=-1).view(-1, 1)
                embed_norms = LA.norm(embeddings, dim=-1)
                for i, token_id in enumerate(scaling_ids):
                    embed_norm = embed_norms[i]
                    self.bias_transformations[token_id] += (embeddings[i].unsqueeze(0).mm(bias_subspaces[:k].T).T * bias_subspaces[:k]  / bias_norms).sum(dim=0) * embed_norm

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

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        bias_transformations = self.bias_transformations
        cpu_input_ids = input_ids.detach().to("cpu")
        transformation_tensors = bias_transformations.gather(dim=0, index=cpu_input_ids.view(-1, 1).repeat_interleave(inputs_embeds.shape[-1], dim=-1))
        transformation_tensors = transformation_tensors.to(device)
        # orig_norms = LA.norm(inputs_embeds, dim=-1)

        inputs_embeds -= transformation_tensors
        # transformed_norms = LA.norm(inputs_embeds, dim=-1)
        # inputs_embeds *= (orig_norms / transformed_norms).view(*inputs_embeds.shape[:2], 1).repeat_interleave(inputs_embeds.shape[-1], dim=-1)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
