import torch.nn as nn

from .baseRNN import BaseRNN


class EncoderRNN(BaseRNN):

    def __init__(self, input_seq_len, input_dim, hidden_dim, input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        super(EncoderRNN, self).__init__(input_seq_len,hidden_dim, input_dropout_p=input_dropout_p,
                                         dropout_p=dropout_p, n_layers=n_layers, rnn_cell=rnn_cell)

        self.input_dim=input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim//2)
        self.relu1=nn.LeakyReLU(0.01)
        self.linear2=nn.Linear(hidden_dim//2,hidden_dim)
        self.relu2=nn.LeakyReLU(0.01)
        self.rnn = self.rnn_cell(hidden_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout_p)

    def forward(self, x):
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu2(x)
        return self.rnn(x)


