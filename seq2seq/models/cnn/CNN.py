import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self,in_features,out_features=3,use_day_ago_info=True):
        super(CNNModel,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.use_day_ago_info=use_day_ago_info
        self.relu=nn.LeakyReLU(0.01)
        self.conv1=nn.Conv1d(in_channels=in_features,out_channels=64,kernel_size=3,padding=1)
        self.conv2=nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        #self.avgpool=nn.AvgPool1d(kernel_size=2,padding=0)
        self.conv3=nn.Conv1d(in_channels=128,out_channels=out_features,kernel_size=3,padding=1)
        self.conv4=nn.Conv1d(in_channels=6,out_channels=3,kernel_size=3,padding=1,groups=3)
        self.linear1=nn.Linear(3,32)
        self.linear2=nn.Linear(32,64)
        self.linear3=nn.Linear(64,3)

    def forward(self,x,one_day_ago_features=None):
        # x->[batch_size,seq_len,feature_dim]
        # one_day_ago_features->[batch_size,seq_len,feature_dim]
        x=torch.transpose(x,1,2)
        if one_day_ago_features is not None:
            one_day_ago_features=torch.transpose(one_day_ago_features,1,2)
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))

        assert one_day_ago_features is None or x.size()==one_day_ago_features.size()
        if one_day_ago_features is not None:
            feature0 = torch.cat((x[:, :1, :], one_day_ago_features[:, :1, :]), -2)
            feature1 = torch.cat((x[:, 1:2, :], one_day_ago_features[:, 1:2, :]), -2)
            feature2 = torch.cat((x[:, 2:, :], one_day_ago_features[:, 2:, :]), -2)
            features = torch.cat((feature0, feature1, feature2), 1)
            x = self.conv4(features)

        x=torch.transpose(x,1,2)
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.linear3(x)



        return x

if __name__=='__main__':
    x=torch.randn(32,108,32)
    one_day_ago=torch.randn(32,108,3)
    model=CNNModel(32,3)
    output=model(x,one_day_ago)
