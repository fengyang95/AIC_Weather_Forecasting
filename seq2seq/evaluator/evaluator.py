from __future__ import print_function,division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Evaluator(object):
    def __init__(self,loss=nn.MSELoss(),batch_size=64,delay=36,valid_feature_indices=list(range(0,3)),begin_compute_loss_index=0):
        self.loss=loss
        self.batch_size=batch_size
        self.delay=delay
        self.valid_feature_indices=valid_feature_indices
        self.begin_compute_loss_index=begin_compute_loss_index

    def evaluate(self,model,data,device=torch.device('cpu')):
        model.eval()
        loss=self.loss
        dataloader= DataLoader(dataset=data,batch_size=self.batch_size,shuffle=True,num_workers=0)
        _loss_val=0.0
        rmse=0
        with torch.no_grad():
            model=model.to(device)
            for batch in dataloader:
                input_vars=batch['X'].to(device)
                decoder_inputs=batch['decoder_inputs'].to(device)
                target_vars=batch['y'].to(device)
                atten_features=batch['atten_features'].to(device)
                seq2seq_outputs=model(input_vars,decoder_inputs,atten_features)
                _loss_val+=loss(seq2seq_outputs[:,self.begin_compute_loss_index:,self.valid_feature_indices],target_vars[:,self.begin_compute_loss_index:,self.valid_feature_indices]).item()

        return _loss_val/(len(dataloader)),rmse

