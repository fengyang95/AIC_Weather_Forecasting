import torch
class Predictor(object):
    def __init__(self,model):
        self.model=model
        self.model.eval()

    def predict(self,src,one_day_ago):
        with torch.no_grad():
            output=self.model(src,one_day_ago)
        return output



