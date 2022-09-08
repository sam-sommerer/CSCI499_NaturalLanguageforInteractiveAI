import torch
from torch.autograd import Variable

# IMPLEMENT YOUR MODEL CLASS HERE


class LSTM(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        input_size,
        hidden_size,
        num_layers,
        seq_length,
        num_actions,
        num_targets,
        vocab_size,
        embedding_dim,
    ):
        super(LSTM, self).__init__()
        self.num_classes = (
            num_classes  # number of classes (should be 2 for our purposes)
        )
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size (batch size?)
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        # self.lstm = torch.nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        # )  # lstm
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )  # lstm
        self.fc_1 = torch.nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc_actions = torch.nn.Linear(
            128, num_actions
        )  # fully connected last layer
        self.fc_targets = torch.nn.Linear(128, num_targets)

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # print(f"x.size(): {x.size()}")
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # internal state
        embeds = self.embedding(x).squeeze()
        # print(f"embeds.shape: {embeds.shape}")
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            embeds, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        # print(f"output.shape: {output.shape}")
        # print(f"hn.shape: {hn.shape}")
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # hn = torch.transpose(hn, 0, 1)
        out = self.relu(hn)
        # print(f"out 1: {out.shape}")
        out = self.fc_1(out)  # first Dense
        # print(f"out 2: {out.shape}")
        out = self.relu(out)  # relu
        # print(f"out 3: {out.shape}")
        out_actions = self.fc_actions(out)  # actions output
        # print(f"out 4: {out.shape}")
        out_targets = self.fc_targets(out)  # targets output
        return self.softmax(out_actions), self.softmax(out_targets)
