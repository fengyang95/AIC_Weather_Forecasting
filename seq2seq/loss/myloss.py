from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
class WFMSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean',delay=36,loss_weights=((0.5,0.5),(1./3,1./3,1./3))):
        super(WFMSELoss, self).__init__(size_average, reduce, reduction)
        self.delay=delay
        self.loss_weights=loss_weights

    def forward(self, input, target):
        loss_overlapping=F.mse_loss(input[:,:-self.delay,:],target[:,:-self.delay,:],reduction=self.reduction)
        loss_predict_t2m=F.mse_loss(input[:,-self.delay:,0],target[:,-self.delay:,0])
        loss_predict_rh2m=F.mse_loss(input[:,-self.delay:,1],target[:,-self.delay:,1])
        loss_predict_w10m=F.mse_loss(input[:,-self.delay:,2],target[:,-self.delay:,2])
        all_loss=self.loss_weights[0][0]*loss_overlapping+self.loss_weights[0][1]*(
            self.loss_weights[1][0]*loss_predict_t2m+
            self.loss_weights[1][1]*loss_predict_rh2m+
            self.loss_weights[1][2]*loss_predict_w10m
        )
        return all_loss

if __name__=='__main__':
    input=torch.randn(5,4,10)
    target=torch.zeros_like(input)
    loss=WFMSELoss(delay=3)
    print(loss(input,target).numpy())