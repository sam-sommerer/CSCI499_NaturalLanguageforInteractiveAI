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

        # self.lstm = torch.nn.LSTM(
        #     input_size=self.embedding_dim,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     batch_first=self.batch_first,
        # )
        encoder_layers = torch.nn.TransformerEncoderLayer(self.embedding_dim, 2, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, self.num_layers)

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # hidden state, initial input into LSTM
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # internal state

        # should be shape (batch_size x instruction_cutoff_len x self.embedding_dim) if batch_first == True ?
        embeds = self.embedding(x).squeeze()

        # # Propagate input through LSTM
        # output, (hn, cn) = self.lstm(
        #     embeds, (h_0, c_0)
        # )  # lstm with input, hidden, and internal state

        output = self.transformer_encoder(x)
        print(f"output.shape: {output.shape}")

        return None


class Decoder(torch.nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        num_actions,
        num_targets,
        batch_first=True,
    ):
        super(Decoder, self).__init__()
        # self.num_actions_targets_misc = num_actions_targets_misc
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.batch_first = batch_first
        self.num_actions_targets_misc = self.num_actions + self.num_targets + 6

        self.lstm = torch.nn.LSTM(
            input_size=self.num_actions_targets_misc,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )

        self.attn = torch.nn.Linear(
            self.num_actions_targets_misc + self.hidden_size,
            96,  # 96 is dim 1 of encoder_outputs
        )
        # self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # num actions should be like 8
        self.actions_fc = torch.nn.Linear(self.hidden_size, self.num_actions + 3)

        # num targets should be like 80
        self.targets_fc = torch.nn.Linear(self.hidden_size, self.num_targets + 3)

    def forward(  # action set and target set we need to add <SOS> to, and <pad>, and <EOS>; create a new dict mapping
        self, x, hidden_state, internal_state, encoder_outputs
    ):  # pass in true labels in here too for teacher forcing?
        """
        The first x to be passed into the decoder should be <SOS> and the last should be <EOS>. hidden_state and
        internal_state should be the hidden and internal states from the previous forward pass.
        """

        # print(f"x.shape: {x.shape}")
        # print(f"hidden_state.shape: {hidden_state.shape}")
        # print(f"internal_state.shape: {internal_state.shape}")

        reshaped_hidden_state = hidden_state.reshape(x.size(0), x.size(1), -1)

        # print(f"reshaped_hidden_state.shape: {reshaped_hidden_state.shape}")

        attn_weights = torch.nn.functional.softmax(
            self.attn(torch.cat((x, reshaped_hidden_state), dim=-1))
        )
        # print(f"attn_weights.shape: {attn_weights.shape}")
        # print(f"encoder_outputs.shape: {encoder_outputs.shape}")
        attn_applied = torch.bmm(
            encoder_outputs.reshape(
                encoder_outputs.size(0), -1, encoder_outputs.size(1)
            ),
            attn_weights.reshape(attn_weights.size(0), -1, 1)
        )
        # print(f"attn_applied.shape: {attn_applied.shape}")
        # weighted_sum = torch.sum(attn_applied, dim=0)
        # print(f"weighted_sum.shape: {weighted_sum.shape}")

        attn_input = attn_applied.reshape(1, attn_applied.size(0), -1)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x.float(), (attn_input, internal_state)
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


class EncoderDecoderTransformer(torch.nn.Module):
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
        decoder_hidden_size,
        decoder_num_layers,
        num_actions,
        num_targets,
        batch_first=True,
        num_predictions=5,
    ):
        super(EncoderDecoderTransformer, self).__init__()

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
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            num_actions=num_actions,
            num_targets=num_targets,
            batch_first=batch_first,
        )

    # what should the format of true labels be? a string in the format "(action, target)"?
    def forward(self, x, true_labels=None, teacher_forcing=False):
        encoder_outputs, hidden_state, internal_state = self.encoder(
            x
        )  # pass output into decoder

        # print(f"x.shape: {x.shape}")

        # h_0 = Variable(
        #     torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # )  # hidden state
        # c_0 = Variable(
        #     torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # )  # internal state

        action_preds = torch.zeros(
            self.num_predictions, x.size(0), self.num_actions + 3
        )
        target_preds = torch.zeros(
            self.num_predictions, x.size(0), self.num_targets + 3
        )

        # action_preds = []
        # target_preds = []

        # decoder inputs should be encoded already, concatenate one-hot vectors representing action and target
        # use the indices for actions and targets
        # decoder_input = "<SOS>"

        sos_actions = torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=self.num_actions + 3
        )
        sos_targets = torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=self.num_targets + 3
        )

        # print(f"sos_actions.shape: {sos_actions.shape}")
        # print(f"sos_targets.shape: {sos_targets.shape}")

        decoder_input = torch.cat((sos_actions, sos_targets), dim=1)
        decoder_input = decoder_input.repeat(x.size(0), 1)  # x.size(0) is batch size
        decoder_input = torch.unsqueeze(decoder_input, 1)

        # print(f"decoder_input.shape: {decoder_input.shape}")

        if teacher_forcing:
            # print(f"true_labels.shape: {true_labels.shape}")
            # print(f"true_labels: {true_labels}")
            for i in range(
                self.num_predictions
            ):  # add <pad> after <EOS> occurs for the true label
                action_pred, target_pred, hidden_state, internal_state = self.decoder(
                    decoder_input, hidden_state, internal_state, encoder_outputs
                )

                # print(f"action_pred.shape: {action_pred.shape}")
                # print(f"target_pred.shape: {target_pred.shape}")

                # action_preds.append(action_pred)
                # target_preds.append(target_pred)

                # action_preds[:, i, :] = action_pred
                # target_preds[:, i, :] = target_pred

                action_preds[i] = action_pred
                target_preds[i] = target_pred

                true_action_idxs, true_target_idxs = (
                    true_labels[:, i, 0],
                    true_labels[:, i, 1],
                )

                # print(f"true_action_idx.shape: {true_action_idxs.shape}")
                # print(true_action_idxs)

                true_actions_one_hots = torch.nn.functional.one_hot(
                    true_action_idxs.long(), num_classes=self.num_actions + 3
                )
                true_targets_one_hots = torch.nn.functional.one_hot(
                    true_target_idxs.long(), num_classes=self.num_targets + 3
                )

                # print(f"true_actions_one_hots.shape: {true_actions_one_hots.shape}")

                decoder_input = torch.cat(
                    (true_actions_one_hots, true_targets_one_hots), dim=1
                )  # concat along dimension 0?
                # print(f"decoder_input.shape: {decoder_input.shape}")
                decoder_input = torch.unsqueeze(decoder_input, 1)
        else:  # student forcing
            for i in range(self.num_predictions):
                action_pred, target_pred, hidden_state, internal_state = self.decoder(
                    decoder_input, hidden_state, internal_state
                )
                action_preds[i] = action_pred
                target_preds[i] = target_pred
                # decoder_input = true_labels[i]

                max_prob_action_idx = torch.topk(action_pred, 1).indices[0]
                max_prob_target_idx = torch.topk(target_pred, 1).indices[0]

                true_action_one_hot = torch.nn.functional.one_hot(
                    torch.tensor([max_prob_action_idx]),
                    num_classes=self.num_actions + 3,
                )
                true_target_one_hot = torch.nn.functional.one_hot(
                    torch.tensor([max_prob_target_idx]),
                    num_classes=self.num_targets + 3,
                )

                decoder_input = torch.cat(
                    (true_action_one_hot, true_target_one_hot)
                )  # concat along dimension 0?

        action_preds = torch.reshape(
            action_preds.float(),
            (x.size(0), self.num_predictions, self.num_actions + 3),
        )
        target_preds = torch.reshape(
            target_preds.float(),
            (x.size(0), self.num_predictions, self.num_targets + 3),
        )

        return action_preds, target_preds
