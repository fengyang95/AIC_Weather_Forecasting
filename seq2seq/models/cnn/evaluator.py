from __future__ import print_function,division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Evaluator(object):
    def __init__(self,loss=nn.MSELoss(),batch_size=64,delay=36):
        self.loss=loss
        self.batch_size=batch_size
        self.delay=delay

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
                #decoder_inputs=batch['decoder_inputs'].to(device)
                #decoder_inputs=batch['X'][:,:,:3].to(device)
                target_vars=batch['y'].to(device)

                day_ago_data=None
                if model.use_day_ago_info:
                    day_ago_data=batch['one_day_ago'].to(device)
                seq2seq_outputs=model(input_vars,day_ago_data)
                _loss_val+=loss(seq2seq_outputs,target_vars).item()

        return _loss_val/(len(dataloader)),rmse

