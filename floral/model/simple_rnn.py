"""
Recurrent Neural Network for Shakespeare Dataset
"""
import math
import torch
import torch.nn as nn


# # https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih_layer = nn.Linear(input_size, 4 * hidden_size)
        self.hh_layer = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(
        self, input: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        gates = self.ih_layer(input) + self.hh_layer(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=512, batch_first=False, num_layers=2):
        super().__init__()
        self.batch_first = batch_first
        self.layers = nn.ModuleList([LSTMLayer(input_size, hidden_size)] + [
            LSTMLayer(hidden_size, hidden_size) for _ in range(num_layers-1)
        ])

    def forward(
        self, input: torch.Tensor, states: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # List[LSTMState]: One state per layer
        output = input
        output_states = []
        if self.batch_first:
            output = output.transpose(0,1)
        for layer, state in zip(self.layers, states):
            output, output_state = layer(output, state)
            output_states.append(output_state)
        if self.batch_first:
            output = output.transpose(0,1)
        return output, output_states


class SimpleRNN(nn.Module):

    def __init__(self, vocab_size=90,
                 embedding_dim=8, hidden_dim=512, num_layers=2):
        super().__init__()

        # set class variables
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_lstm_layers = num_layers
        # self.lstm = nn.LSTM(
        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.hidden = None

    def forward(self, input, hidden=None):
        # if hidden is None:
        #     if self.hidden is None:
        #         self.hidden = self.init_hidden(input.size(0), input.device)
        #     hidden = self.hidden
        hidden = self.init_hidden(input.size(0), input.device)
        embeds = self.embedding(input)
        lstm_out, _ = self.lstm(embeds, hidden)
        out = self.fc(lstm_out)
        # flatten the output
        out = out.reshape(-1, self.vocab_size)
        return out

    def init_hidden(self, batch_size, device):
        hidden = (torch.zeros(self.num_lstm_layers, batch_size,
                              self.hidden_dim).to(device),
                  torch.zeros(self.num_lstm_layers, batch_size,
                              self.hidden_dim).to(device))
        return hidden


def simple_rnn(pretrained=False, num_classes=90):
    return SimpleRNN(vocab_size=num_classes)


def mini_simple_rnn(pretrained=False, num_classes=90):
    return SimpleRNN(vocab_size=num_classes, hidden_dim=128)
