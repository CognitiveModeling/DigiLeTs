import torch
import torch.nn as nn

class Encoding(nn.Module):
    """
    Split Feedforward (Participant and Character) -> LSTM model proposed in our paper
    """
    def __init__(self, character_size, participant_size, embedding_size, hidden_size, num_layers,
                 hidden_bias, dropout, output_size, output_bias, inf, infer_target):
        super(Encoding, self).__init__()
        self.linear_character = nn.Linear(
            in_features=character_size,
            out_features=hidden_size,
            bias=output_bias)
        self.linear_participant = nn.Linear(
            in_features=participant_size,
            out_features=embedding_size,
            bias=output_bias)
        self.lstm = nn.LSTM(
            input_size=hidden_size+embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=hidden_bias)
        self.linear2 = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias=output_bias)
        # one-shot inf mechanism: only adapt weights into first ff layer
        self.set_infer(inf, infer_target)

    def set_infer(self, inf, infer_target):
        """
        Method to set which parts of the net weights are backpropagated to.
        If inf is true weights are only backpropagated onto the linear layers.
        :param inf: Whether to backpropagate gradients onto only the linear layer
        :param infer_target: "character", "participant" or "both", to decide which linear layer the weights should be
            backpropagated to
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

        if infer_target == "character":
            for p in self.linear_character.parameters():
                p.requires_grad = True
            for p in self.linear_participant.parameters():
                p.requires_grad = False
        elif infer_target == "participant":
            for p in self.linear_character.parameters():
                p.requires_grad = False
            for p in self.linear_participant.parameters():
                p.requires_grad = True
        elif infer_target == "both":
            for p in self.linear_character.parameters():
                p.requires_grad = True
            for p in self.linear_participant.parameters():
                p.requires_grad = True


    def forward(self, character, participant, lengths):
        x1 = self.linear_character(character)
        x2 = self.linear_participant(participant)
        x = torch.cat((x1, x2), dim=2)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, enforce_sorted=False)
        x, (h, c) = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = self.linear2(x)
        return x

