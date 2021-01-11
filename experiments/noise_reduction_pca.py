import numpy as np
import scipy
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from constants import GENDER_PAIRS, MALE_TERMS, FEMALE_TERMS
from utils import find_embedding_layer


class NoiseReductionPCA(object):
    def fit(self, data):
        n = data.shape[0]
        d = data.shape[1]
        r = min(n - 2, d)

        data  -= data.mean(axis=0)
        std_div = np.cov(data)

        eig_vals, eig_vecs = np.linalg.eigh(std_div)

        eig_vals, eig_vecs = eig_vals[::-1], eig_vecs[::-1]
        dual_vals = eig_vals[:r]
        dual_vecs = eig_vecs[:r, :]

        cum_dual_vals = dual_vals.cumsum()
        nrm_vals = dual_vals - ((np.diag(std_div).sum() - cum_dual_vals) / np.arange(n-2, n-r-2, -1))
        nrm_vecs = (dual_vecs @ data).T / np.sqrt((n - 1) * np.abs(nrm_vals))
        nrm_scores = dual_vecs.T * np.sqrt(n * np.abs(nrm_vals))

        self.nrm_vals_ = nrm_vals
        self.components_ = nrm_vecs.T
        self.nrm_vecs_ = nrm_vecs
        self.nrm_scores_ = nrm_scores

    def scores(self, embeddings, n_components = 0):
        return (embeddings @ self.nrm_vecs_)[:, n_components]
        

# if __name__ == "__main__":
#     pca = NoiseReductionPCA()
# 
#     config = AutoConfig.from_pretrained('bert-base-uncased')
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', config=config)
#     model = AutoModel.from_pretrained('bert-base-uncased', config=config)
# 
#     embeddings = find_embedding_layer(model)
#     matrix = []
#     word_sets = [MALE_TERMS, FEMALE_TERMS]
#     for word_set in word_sets:
#         word_ids = tokenizer.convert_tokens_to_ids(word_set)
#         target_embeddings = embeddings(torch.tensor(word_ids))
#         center = target_embeddings.mean(dim=0)
#         matrix.extend((target_embeddings - center).detach())
#     centered_matrix = torch.stack(matrix).numpy()
# 
#     words = sum(word_sets, [])
#     word_ids = tokenizer.convert_tokens_to_ids(words)
#     ord_matrix = embeddings(torch.tensor(word_ids)).detach().numpy()
# 
#     pca.fit(ord_matrix)
#     import ipdb; ipdb.set_trace()
# 
