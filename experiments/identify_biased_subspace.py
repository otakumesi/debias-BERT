from absl import app
from absl import flags

from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from sklearn.decomposition import PCA
import torch.linalg as LA

from utils import find_embedding_layer
from constants import SETS_LIST

FLAGS = flags.FLAGS

flags.DEFINE_bool("all_vocab_space", False, "output all vocab space")
flags.DEFINE_string("freq_spaces", None, "output all vocab space")
flags.DEFINE_string("model_name", "bert-base-uncased", "model name in transformers")
flags.DEFINE_string("output_file", None, "output file name")
flags.DEFINE_string("wordset_file", None, "word sets with related to bias")
flags.DEFINE_integer("n_components", 10, "component number of PCA")


class BiasSubSpaceIdentifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.freq_biases = None

    def identify_freq_bias(self, n_components=10):
        embeddings = find_embedding_layer(self.model)

        word_ids = range(0, self.tokenizer.vocab_size)
        target_embeds = embeddings(torch.tensor(word_ids))
        return self.__do_pca(target_embeds, n_components)

    def identify_bias(self, word_set, n_components=10):
        embeddings = find_embedding_layer(self.model)

        word_set = set(word_set)
        word_ids = self.tokenizer.convert_tokens_to_ids(word_set)

        target_embeds = embeddings(torch.tensor(word_ids))
        return self.__do_pca(target_embeds, n_components)

    def identify_bias_between_word_sets(self, social_group_word_sets, n_components=10, freq_spaces=None, k=2):
        embeddings = find_embedding_layer(self.model)

        matrix = []
        for word_set in social_group_word_sets:
            word_ids = self.tokenizer.convert_tokens_to_ids(word_set)
            target_embeddings = embeddings(torch.tensor(word_ids))
            if freq_spaces:
                freq_subspaces = torch.load(freq_spaces)
                freq_subspaces = torch.from_numpy(freq_subspaces).float()[:k]
                freq_norms = LA.norm(freq_subspaces, dim=-1).view(-1, 1)
                embed_norms = LA.norm(target_embeddings, dim=-1)

                freq_subspaces = (((target_embeddings.mm(freq_subspaces.T)) * freq_subspaces / freq_norms).T * embed_norms).T
                target_embeddings -= freq_subspaces
            center = target_embeddings.mean(dim=0)
            matrix.extend((target_embeddings - center).detach())

        matrix = torch.stack(matrix)
        return self.__do_pca(matrix, n_components)

    def __do_pca(self, target_embeds, n_components):
        matrix = target_embeds.detach().numpy()

        pca = PCA()
        pca.fit(matrix)

        return pca.components_[:n_components]


def main(argv):
    print("--- load models ---")
    config = AutoConfig.from_pretrained(FLAGS.model_name)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name, config=config)
    model = AutoModel.from_pretrained(FLAGS.model_name, config=config)
    identifier = BiasSubSpaceIdentifier(model=model, tokenizer=tokenizer)

    print("--- identify subspace ---")
    subspaces = []
    if not FLAGS.all_vocab_space:
        for word_set in SETS_LIST:
            subspace = identifier.identify_bias_between_word_sets(word_set, n_components=FLAGS.n_components, freq_spaces=FLAGS.freq_spaces)
            # subspace = identifier.identify_bias(sum(word_set, []), n_components=FLAGS.n_components)
            print(subspace.shape)
            subspaces.append(torch.from_numpy(subspace))
        subspaces = torch.stack(subspaces)

        torch.save(subspaces, FLAGS.output_file)
    else:
        subspace = identifier.identify_freq_bias(n_components=FLAGS.n_components)
        torch.save(subspace, FLAGS.output_file)

    print("--- save subspace ---")


if __name__ == "__main__":
    app.run(main)
