import random
import numpy as np
import torch
import torch.nn as nn

from .baseRNN import BaseRNN
from .attention import Attention

class DecoderRNN(BaseRNN):
    def __init__(self, input_seq_len, output_seq_len, output_dim, hidden_dim, n_layers=1, rnn_cell='gru',
                 bidirectional=False, input_dropout_p=0, dropout_p=0,use_attention=False):
        super(DecoderRNN,self).__init__(input_seq_len,
                                        hidden_dim=hidden_dim, input_dropout_p=input_dropout_p,
                                        dropout_p=dropout_p, n_layers=n_layers, rnn_cell=rnn_cell)
        self.output_seq_len=output_seq_len
        self.output_dim=output_dim
        self.bidirecitonal_encoder=bidirectional
        self.use_attention=use_attention
        self.rnn=self.rnn_cell(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_p)
        if self.use_attention:
            self.attention=Attention(self.hidden_dim)
        self.out=nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,input,hidden,encoder_outputs):
        batch_size=input.size(0)
        output_dim=self.output_dim
        output=self.input_dropout(input)
        hidden=self._init_state(hidden)
        output,hidden=self.rnn(output,hidden)
        if self.use_attention:
            output,atten=self.attention(output,encoder_outputs)
        output=self.out(output.contiguous().view(-1,self.hidden_dim))
        output=output.view(batch_size,-1,output_dim)
        return output,hidden


    def _init_state(self,encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden,tuple):
            encoder_hidden=tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden=self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self,h):
        if self.bidirecitonal_encoder:
            h=torch.cat([h[0:h.size(0):2],h[1:h.size(0):2]],2)
        return h
