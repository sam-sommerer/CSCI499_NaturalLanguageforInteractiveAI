import torch.nn as nn


class SkipgramModel(nn.Module):
    def __init__(self, vocab_size=3000, embedding_dim=16):
        super(SkipgramModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        self.fc1 = nn.Linear(self.embedding_dim, self.vocab_size)  # output layer

    def forward(self, x):
        embeddings = self.embedding(x).squeeze()

        return self.fc1(embeddings)
