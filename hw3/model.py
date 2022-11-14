# IMPLEMENT YOUR MODEL CLASS HERE

import torch
from torch.autograd import Variable


class Encoder(torch.nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_size, num_layers, batch_first=True
    ):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # hidden state, initial input into LSTM
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # internal state

        # should be shape (batch_size x instruction_cutoff_len x self.embedding_dim) if batch_first == True
        embeds = self.embedding(x).squeeze()

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            embeds, (h_0, c_0)
        )  # lstm with input, hidden, and internal state

        return output, hn, cn


class Decoder(torch.nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self):
        pass

    def forward(self, x):  # pass in true labels in here too for teacher forcing?
        pass


class EncoderDecoder(torch.nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_size, num_layers, batch_first=True
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        output, hn, cn = self.encoder(x)  # pass output into decoder

