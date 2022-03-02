__author__ = "Julius Wührer"

import torch.nn as nn


class LSTM(nn.Module):
    """
    Simple Feedforward -> LSTM module used by Fabi et al. (2020).
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 hidden_bias, dropout, output_size, output_bias, inf):
        super(LSTM, self).__init__()
        self.linear1 = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
            bias=output_bias)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=hidden_bias)
        self.linear2 = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias=output_bias)
        # one-shot inf mechanism: only adapt weights into first ff layer
        self.set_infer(inf)

    def set_infer(self, inf):
        """
        Method to set which parts of the net gradients are backpropagated to.
        If inf is true gradients are only backpropagated onto the linear layer.
        :param inf: Whether to backpropagate gradients onto only the linear layer
        """
        if inf == True:
            for p in self.lstm.parameters():
                p.requires_grad = False
            for q in self.linear2.parameters():
                q.requires_grad = False
        else:
            for p in self.lstm.parameters():
                p.requires_grad = True
            for q in self.linear2.parameters():
                q.requires_grad = True

        for p in self.linear1.parameters():
            p.requires_grad = True

    def forward(self, x, _, lengths):
        x = self.linear1(x)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, enforce_sorted=False)
        x, (h, c) = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = self.linear2(x)
        return x

