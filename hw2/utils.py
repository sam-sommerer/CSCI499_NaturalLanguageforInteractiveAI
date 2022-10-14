import json
import gensim
import tqdm
import torch


def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with gensim.utils.open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embed.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )


def convert_indices_to_multihot(indices_batch, output_shape):
    word_indices = []

    for i, indices in enumerate(indices_batch):
        for index in indices:
            word_indices.append([i, index])

    word_indices_tensor = torch.tensor(word_indices)

    multihot_vectors = torch.zeros(output_shape)
    multihot_vectors[word_indices_tensor[:, 0], word_indices_tensor[:, 1]] += 1

    return multihot_vectors
