import torch
class Predictor(object):
    def __init__(self,model):
        self.model=model
        self.model.eval()

    def predict(self,src,decoder_inputs,attenfeatures):
        with torch.no_grad():
            output=self.model(src,decoder_inputs,attenfeatures)
        return output



