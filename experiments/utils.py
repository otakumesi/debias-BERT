from torch.nn import Embedding
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


def find_embedding_layer(model):
    for module in model.modules():
        if isinstance(module, BertEmbeddings):
            return module.word_embeddings
        if isinstance(module, GPT2Model):
            return module.wte

    for module in model.modules():
        if isinstance(module, Embedding):
            return module

    raise RuntimeError('Not Found Embedding module!')
