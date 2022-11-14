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

        # should be shape (batch_size x instruction_cutoff_len x self.embedding_dim) if batch_first == True ?
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

    def __init__(
        self, output_size, embedding_dim, hidden_size, num_layers, num_actions, num_targets, batch_first=True
    ):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.batch_first = batch_first

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.output_size, embedding_dim=self.embedding_dim
        )

        self.action_lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

        self.target_lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

        self.actions_fc = torch.nn.Linear(self.hidden_size, self.num_actions)
        self.targets_fc = torch.nn.Linear(self.hidden_size, self.num_targets)

        self.softmax = torch.nn.LogSoftmax(dim=0)  # not sure what dim is supposed to be here

    def forward(
        self, x, action_hidden_state, action_internal_state, target_hidden_state, target_internal_state
    ):  # pass in true labels in here too for teacher forcing?
        """
        The first x to be passed into the decoder should be <SOS> and the last should be <EOS>. hidden_state and
        internal_state should be the hidden and internal states from the previous forward pass.
        """

        # These should be created in the training loop and passed in during the first forward pass
        # h_0 = Variable(
        #     torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # )  # hidden state, initial input into LSTM
        # c_0 = Variable(
        #     torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # )  # internal state

        embeds = self.embedding(x).squeeze()

        # Propagate input through LSTM
        action_output, (action_hn, action_cn) = self.action_lstm(
            embeds, (action_hidden_state, action_internal_state)
        )  # lstm with input, hidden, and internal state

        # Propagate input through LSTM
        target_output, (target_hn, target_cn) = self.target_lstm(
            embeds, (target_hidden_state, target_internal_state)
        )  # lstm with input, hidden, and internal state

        action_pred = self.softmax(self.actions_fc(action_output))
        target_pred = self.softmax(self.targets_fc(target_output))

        return action_pred, action_output, target_pred, target_output


class EncoderDecoder(torch.nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(
        self,
        vocab_size,
        encoder_embedding_dim,
        encoder_hidden_size,
        encoder_num_layers,
        output_size,
        decoder_embedding_dim,
        decoder_hidden_size,
        decoder_num_layers,
        batch_first=True,
        num_predictions=5
    ):
        super(EncoderDecoder, self).__init__()

        self.N = num_predictions

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=encoder_embedding_dim,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            batch_first=batch_first,
        )
        self.decoder = Decoder(
            output_size=output_size,
            embedding_dim=decoder_embedding_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            batch_first=batch_first,
        )

    def forward(self, x, true_labels=None, teacher_forcing=False):
        output, hn, cn = self.encoder(x)  # pass output into decoder


