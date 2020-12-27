from absl import app
from absl import flags

from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from sklearn.decomposition import PCA
import inflect

from utils import find_embedding_layer


p = inflect.engine()
FLAGS = flags.FLAGS

flags.DEFINE_string("model_name", "bert-base-uncased",
                    "model name in transformers")
flags.DEFINE_string("output_file", None, "output file name")
flags.DEFINE_string("wordset_file", None, "word sets with related to bias")
flags.DEFINE_integer("n_components", 10, "component number of PCA")


GENDER_PAIRS = [["woman", "man"],
                ["girl", "boy"],
                ["she", "he"],
                ["mother", "father"],
                ["sister", "brother"],
                ["daughter", "son"],
                ["female", "male"],
                ["her", "his"],
                ["her", "him"],
                ["herself", "himself"],
                ["wife", "husband"],
                ["female", "masculine"],
                ['mrs', 'mr']]


class BiasSubSpaceIdentifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def identify(self, social_group_word_sets, n_components=10):
        matrix = []
        embeddings = find_embedding_layer(self.model)
        for eq_word_set in social_group_word_sets:
            word_ids = self.tokenizer.convert_tokens_to_ids(eq_word_set)
            target_embeddings = embeddings(torch.tensor(word_ids))
            center = target_embeddings.mean(dim=0)
            matrix.extend((target_embeddings - center).detach())

        matrix = torch.stack(matrix).numpy()
        pca = PCA(n_components=n_components)
        pca.fit(matrix)

        return pca.components_[0]


def main(argv):
    print("--- load models ---")
    config = AutoConfig.from_pretrained(FLAGS.model_name)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name, config=config)
    model = AutoModel.from_pretrained(FLAGS.model_name, config=config)
    identifier = BiasSubSpaceIdentifier(model=model, tokenizer=tokenizer)

    if FLAGS.wordset_file is None:
        word_sets = [[p.plural(f), p.plural(m)] for f, m in GENDER_PAIRS if f in ["mrs", "her", "she"]] + GENDER_PAIRS
    else:
        with open(FLAGS.wordset_file) as f:
            word_sets = f.read()

    print("--- identify subspace ---")
    subspace = identifier.identify(word_sets, n_components=FLAGS.n_components)
    subspace = torch.from_numpy(subspace)

    print("--- save subspace ---")
    torch.save(subspace, FLAGS.output_file)


if __name__ == "__main__":
    app.run(main)
