# IMPLEMENT YOUR MODEL CLASS HERE

import torch
import numpy as np
from torch.autograd import Variable

#  For transformer you could just replace all the encoder with transformer
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
        self,
        output_size,
        hidden_size,
        num_layers,
        num_actions,
        num_targets,
        batch_first=True,
    ):
        super(Decoder, self).__init__()
        self.output_size = output_size
        # self.num_actions_targets_misc = num_actions_targets_misc
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.batch_first = batch_first
        self.num_actions_targets_misc = self.num_actions + self.num_targets + 6

        # self.embedding = torch.nn.Embedding(
        #     num_embeddings=self.output_size, embedding_dim=self.embedding_dim
        # )

        self.lstm = torch.nn.LSTM(
            input_size=self.num_actions_targets_misc,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

        # self.target_lstm = torch.nn.LSTM(
        #     input_size=self.embedding_dim,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     batch_first=self.batch_first,
        # )

        # num actions should be like 8
        self.actions_fc = torch.nn.Linear(self.hidden_size, self.num_actions + 3)

        # num targets should be like 80
        self.targets_fc = torch.nn.Linear(self.hidden_size, self.num_targets + 3)

        # self.softmax = torch.nn.LogSoftmax(
        #     dim=0
        # )  # DON'T DO SOFTMAX JUST RETURN LOGITS, crossentropy does softmax

    def forward(  # action set and target set we need to add <SOS> to, and <pad>, and <EOS>; create a new dict mapping
        self, x, hidden_state, internal_state
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

        # embeds = self.embedding(x).squeeze()

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (hidden_state, internal_state)
        )  # lstm with input, hidden, and internal state

        reshaped_hn = hn.view(-1, self.hidden_size)
        # reshaped_cn = cn.view(-1, self.hidden_size)

        # # Propagate input through LSTM
        # target_output, (target_hn, target_cn) = self.target_lstm(
        #     embeds, (target_hidden_state, target_internal_state)
        # )  # lstm with input, hidden, and internal state

        # action_pred = self.softmax(self.actions_fc(reshaped_hn))
        # target_pred = self.softmax(self.targets_fc(reshaped_hn))

        action_pred = self.actions_fc(reshaped_hn)
        target_pred = self.targets_fc(reshaped_hn)

        return action_pred, target_pred, hn, cn


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
        decoder_hidden_size,
        decoder_num_layers,
        num_actions,
        num_targets,
        batch_first=True,
        num_predictions=5,
    ):
        super(EncoderDecoder, self).__init__()

        self.num_actions = num_actions
        self.num_targets = num_targets
        self.num_predictions = num_predictions

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=encoder_embedding_dim,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            batch_first=batch_first,
        )
        self.decoder = Decoder(
            output_size=output_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            num_actions=num_actions,
            num_targets=num_targets,
            batch_first=batch_first,
        )

    # what should the format of true labels be? a string in the format "(action, target)"?
    def forward(self, x, index_to_actions, index_to_targets, true_labels=None, teacher_forcing=False):
        output, hidden_state, internal_state = self.encoder(x)  # pass output into decoder

        # h_0 = Variable(
        #     torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # )  # hidden state
        # c_0 = Variable(
        #     torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # )  # internal state

        action_preds = []
        target_preds = []

        # decoder inputs should be encoded already, concatenate one-hot vectors representing action and target
        # use the indices for actions and targets
        # decoder_input = "<SOS>"

        sos_actions = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=self.num_actions + 3)
        sos_targets = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=self.num_targets + 3)
        decoder_input = torch.cat((sos_actions, sos_targets))

        if teacher_forcing:
            for i in range(self.num_predictions):  # add <pad> after <EOS> occurs for the true label
                action_pred, target_pred, hidden_state, internal_state = self.decoder(
                    decoder_input, hidden_state, internal_state
                )
                action_preds.append(action_pred)
                target_preds.append(target_pred)

                true_action_idx, true_target_idx = true_labels[i]

                true_action_one_hot = torch.nn.functional.one_hot(torch.tensor([true_action_idx]), num_classes=self.num_actions + 3)
                true_target_one_hot = torch.nn.functional.one_hot(torch.tensor([true_target_idx]),
                                                                  num_classes=self.num_targets + 3)

                decoder_input = torch.cat((true_action_one_hot, true_target_one_hot))  # concat along dimension 0?
        else:
            for i in range(self.num_predictions):
                action_pred, target_pred, hidden_state, internal_state = self.decoder(
                    decoder_input, hidden_state, internal_state
                )
                action_preds.append(action_pred)
                target_preds.append(target_pred)
                # decoder_input = true_labels[i]

                max_prob_action_idx = torch.topk(action_pred, 1).indices[0]
                max_prob_target_idx = torch.topk(target_pred, 1).indices[0]

                decoder_input = str((index_to_actions[max_prob_action_idx], index_to_targets[max_prob_target_idx]))

        return action_preds, target_preds



