import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import argparse
import logging
import torch
from torchvision import transforms
from seq2seq.trainer.supervised_trainer import SupervisedTrainer
from seq2seq.models.DecoderRNN import DecoderRNN
from seq2seq.models.EncoderRNN import EncoderRNN
from seq2seq.models.seq2seq import Seq2Seq
from seq2seq.dataset.WFDataset import  WFDataset,ToTensor
from util.checkpoint import Checkpoint
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from seq2seq.optim.optim import Optimizer
raw_input = input  # Python 3
import pandas as pd
import numpy as np
import util.Scaler as Scaler
import util.StdScaler as StdScaler
from seq2seq.loss.myloss import WFMSELoss


parser = argparse.ArgumentParser()
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiement_default',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--use_attention',dest='use_attention',default=False)
parser.add_argument('--rnn_cell_type',dest='rnn_cell_type',default='lstm')
parser.add_argument('--rnn_layers',dest='rnn_layers',default=5,type=int)
parser.add_argument('--rnn_dropout',dest='rnn_dropout',default=0,type=float)
parser.add_argument('--batch_size',default=16,dest='batch_size',type=int)
parser.add_argument('--device',default='cpu',dest='device')
parser.add_argument('--lr',default=1e-4,dest='lr',type=float)
parser.add_argument('--weight_decay',default=5e-4,dest='weight_decay',type=float)
parser.add_argument('--use_custome_loss',dest='use_custome_loss',default=False,type=bool)
parser.add_argument('--scaler',dest='scaler',default='minmax',type=str)
parser.add_argument('--begin_compute_loss_index',dest='begin_compute_loss_index',default=0,type=int)

opt = parser.parse_args()


LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

feature_columns=['t2m_obs', 'rh2m_obs', 'w10m_obs', 't2m_prophet','rh2m_prophet','w10m_prophet',
                 't2m_M','rh2m_M','w10m_M','hour_sin','hour_cos','month_sin','month_cos',
                 'psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs',
                 'v10m_obs', 'RAIN_obs','psfc_M', 'q2m_M', 'd10m_M', 'u10m_M', 'v10m_M',
                 'SWD_M', 'GLW_M', 'HFX_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M',
                 'TC850_M', 'TC700_M', 'TC500_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M',
                 'location_90001','location_90002','location_90003','location_90004',
                 'location_90005','location_90006','location_90007','location_90008',
                 'location_90009','location_90010'] # remove LH_M
print(len(feature_columns))
device = torch.device(opt.device)
train_array_list=[]
validation_array_list=[]

if opt.scaler=='minmax':
    scaler=Scaler.Scaler(Scaler.MINMAX_DICT,feature_range=(-1,1))
else:
    scaler=StdScaler.StdScaler(StdScaler.MEANSTD_DICT)

for i in range(90001,90011):
    train_data = pd.read_csv('../data/train/merge/merged_' + str(i) + '.csv', index_col=0)
    train_data.index=pd.to_datetime(train_data.index)
    train_data['hour']=train_data.index.hour
    train_data['month']=train_data.index.month
    hour_period = 24 / (2 * np.pi)
    train_data['hour_cos']=np.cos(train_data.index.hour / hour_period)
    train_data['hour_sin']=np.sin(train_data.index.hour / hour_period)

    month_period=12/(2*np.pi)
    train_data['month_cos']=np.cos(train_data.index.month / month_period)
    train_data['month_sin']=np.sin(train_data.index.month / month_period)
    #分析相关性发现Q*特征与t2m_obs,rh2m_obs及 w10m的相关性都较小
    for j in range(90001,90011):
        train_data['location_'+str(j)]=[0.]*len(train_data)
    train_data['location_'+str(i)]=[1.]*len(train_data)

    train_data = pd.DataFrame(train_data, columns=feature_columns)

    for col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())


    train_float_data = scaler.transform(train_data)
    train_array_list.append(train_float_data)

    validation_data = pd.read_csv('../data/val/merge/merged_' + str(i) + '.csv', index_col=0)
    validation_data.index = pd.to_datetime(validation_data.index)
    validation_data['hour'] = validation_data.index.hour
    validation_data['month'] = validation_data.index.month
    hour_period = 24 / (2 * np.pi)
    validation_data['hour_cos'] = np.cos(validation_data.index.hour / hour_period)
    validation_data['hour_sin'] = np.sin(validation_data.index.hour / hour_period)

    month_period = 12 / (2 * np.pi)
    validation_data['month_cos'] = np.cos(validation_data.index.month / month_period)
    validation_data['month_sin'] = np.sin(validation_data.index.month / month_period)

    for j in range(90001,90011):
        validation_data['location_'+str(j)]=[0.]*len(validation_data)
    validation_data['location_'+str(i)]=[1.]*len(validation_data)

    validation_data = pd.DataFrame(validation_data,
                                   columns=feature_columns)

    for col in validation_data.columns:
        validation_data[col] = validation_data[col].fillna(validation_data[col].mean())

    validation_float_data = scaler.transform(validation_data)
    validation_array_list.append(validation_float_data)


seqence_len = 72
output_dim=3
delay = 36


if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model.to(device)
else:
    # Prepare dataset
    train = WFDataset(train_array_list,delay=delay,seq_len=seqence_len,outdim=3,transform=transforms.Compose([ToTensor()]))
    dev = WFDataset(validation_array_list,delay=delay,seq_len=seqence_len,outdim=3,transform=transforms.Compose([ToTensor()]),begin_index=100)
    if opt.use_custome_loss:
        loss = WFMSELoss(delay=36, loss_weights=((0.25, 0.75), (0.3, 0.5, 0.2)))
    else:
        loss = nn.MSELoss()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        input_dim=38+10
        hidden_dim=128
        bidirectional = False
        encoder = EncoderRNN(input_seq_len=seqence_len, input_dim=input_dim,hidden_dim=hidden_dim, bidirectional=bidirectional, n_layers=opt.rnn_layers, rnn_cell=opt.rnn_cell_type,dropout_p=opt.rnn_dropout)
        decoder = DecoderRNN(input_seq_len=seqence_len,output_seq_len=delay, output_dim=output_dim, hidden_dim=hidden_dim * 2 if bidirectional else hidden_dim,
                             dropout_p=opt.rnn_dropout,  bidirectional=bidirectional, n_layers=opt.rnn_layers, rnn_cell=opt.rnn_cell_type,use_attention=opt.use_attention)
        seq2seq = Seq2Seq(encoder,decoder,decode_function=torch.tanh).to(device)

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08,0.08)
    print('model:',seq2seq)
    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                          checkpoint_every=1000,
                          print_every=100, expt_dir=opt.expt_dir,device=device,valid_feature_indices=list(range(1,2)),begin_compute_loss_index=opt.begin_compute_loss_index)
    optimizer=Optimizer(optim.Adam(seq2seq.parameters(), lr=opt.lr, weight_decay=opt.weight_decay), max_grad_norm=5)
    #scheduler = StepLR(optimizer.optimizer, 30,gamma=0.5)
    #optimizer.set_scheduler(schedule
    seq2seq = t.train(seq2seq, train,
                      num_epochs=800, dev_data=dev,lr=opt.lr,
                      optimizer=optimizer,
                      resume=opt.resume)
