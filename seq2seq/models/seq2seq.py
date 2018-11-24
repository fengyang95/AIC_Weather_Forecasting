import torch.nn as nn
import torch

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,decode_function=torch.tanh):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.decode_function=decode_function
        self.linear1=nn.Linear(10,self.decoder.hidden_dim//2)
        self.relu1=nn.LeakyReLU(0.01)
        self.linear2=nn.Linear(self.decoder.hidden_dim//2,self.decoder.hidden_dim)
        self.relu2=nn.LeakyReLU(0.01)
        self.final_conv=nn.Conv1d(in_channels=9,out_channels=3,kernel_size=3,padding=1,groups=3)
        self.final_linear=nn.Linear(3,3)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_vars, decoder_inputs=None,atten_features=None):
        encoder_outputs,encoder_hidden=self.encoder(input_vars)

        #print('decoder inputs:')
        #print('min:',np.min(decoder_inputs.detach().cpu().numpy()))
        #print('max:',np.max(decoder_inputs.detach().cpu().numpy()))
        if decoder_inputs is None:
            decoder_inputs=torch.zeros_like(encoder_outputs)
        else:
            decoder_inputs=self.linear1(decoder_inputs)
            decoder_inputs=self.relu1(decoder_inputs)
            decoder_inputs=self.linear2(decoder_inputs)
            decoder_inputs=self.relu2(decoder_inputs)

        result=self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
        output=result[0]

        if atten_features is not None:
            assert atten_features.size(2)==6
            feature0=torch.cat((output[:,:,:1],atten_features[:,:,:2]),-1)
            feature1=torch.cat((output[:,:,1:2],atten_features[:,:,2:4]),-1)
            feature2=torch.cat((output[:,:,2:3],atten_features[:,:,4:6]),-1)
            features=torch.cat((feature0,feature1,feature2),-1)
            output=torch.transpose(self.final_conv(torch.transpose(features,1,2)),1,2)
        #output=self.decode_function(output)
        output=self.final_linear(output)
        return output

