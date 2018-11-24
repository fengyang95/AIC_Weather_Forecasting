import torch.nn as nn

class BaseRNN(nn.Module):
    def __init__(self, input_seq_len, hidden_dim, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN,self).__init__()
        self.input_seq_len=input_seq_len
        self.hidden_dim=hidden_dim
        self.n_layers=n_layers
        self.input_dropout_p=input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.dropout_p=dropout_p
        if rnn_cell.lower()=='lstm':
            self.rnn_cell=nn.LSTM
        elif rnn_cell.lower()=='gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: {0}'.format(rnn_cell))

    def forward(self,*args,**kwargs):
        raise NotImplementedError

