import os
import torch
from seq2seq.evaluator.predictor import Predictor
from util.checkpoint import Checkpoint
raw_input = input  # Python 3
import pandas as pd
import numpy as np
import util.Scaler as Scaler
import util.StdScaler as StdScaler

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

xgbr_X_indices=list(range(6,9))+list(range(19,38))
xgbr_y_indices=list(range(0,3))

print(len(feature_columns))

use_minmax_scaler=True
device=torch.device('cpu')
if use_minmax_scaler:
    scaler = Scaler.Scaler(Scaler.MINMAX_DICT, feature_range=(-1, 1))
else:
    scaler = StdScaler.StdScaler(StdScaler.MEANSTD_DICT)


def get_predict_values(seq_tensor,decoder_input_tensor,anen_tensor,predictor,dst_feature_index):
    pred = predictor.predict(seq_tensor, decoder_input_tensor, anen_tensor).cpu().numpy()[0]
    return pred[:,dst_feature_index]

from datetime import datetime

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

begin_dates=datelist('20181025','20181031')
dst_dates=datelist('20181028','20181103')
end_dates=datelist('20181029','20181104')

data_dir='../data/testb7/merge'
dst_dir='../result/seq2seq_feature48'
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

if __name__=='__main__':
    seqence_len=72
    output_dim=3
    delay=36
    t2m_checkpoint_path=os.path.join('../checkpoints/seq72_feature48_global_t2m_best')
    rh2m_checkpoint_path=os.path.join('../checkpoints/seq72_feature48_global_rh2m_best')
    w10m_checkpoint_path=os.path.join('../checkpoints/seq72_feature48_global_w10m_best')

    t2m_checkpoint=Checkpoint.load(t2m_checkpoint_path)
    rh2m_checkpoint=Checkpoint.load(rh2m_checkpoint_path)
    w10m_checkpoint=Checkpoint.load(w10m_checkpoint_path)

    t2m_predictor=Predictor(t2m_checkpoint.model.to(device))
    rh2m_predictor=Predictor(rh2m_checkpoint.model.to(device))
    w10m_predictor=Predictor(w10m_checkpoint.model.to(device))


    foretimes=37

    for begin_date,dst_date,end_date in zip(begin_dates,dst_dates,end_dates):
        submit_csv = None
        end_date=end_date+' 12-00-00'
        for i in range(90001, 90011):
            df = pd.read_csv(os.path.join(data_dir,'merged_' + str(i) + '.csv'), index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df[begin_date:end_date]
            df['hour'] = df.index.hour
            df['month'] = df.index.month
            hour_period = 24 / (2 * np.pi)
            df['hour_cos'] = np.cos(df.index.hour / hour_period)
            df['hour_sin'] = np.sin(df.index.hour / hour_period)

            month_period = 12 / (2 * np.pi)
            df['month_cos'] = np.cos(df.index.month / month_period)
            df['month_sin'] = np.sin(df.index.month / month_period)
            for j in range(90001, 90011):
                df['location_' + str(j)] = [0.] * len(df)
            df['location_' + str(i)] = [1.] * len(df)
            df = pd.DataFrame(df, columns=feature_columns)

            for col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
            float_data = scaler.transform(df)

            index_list = []
            for j in range(37):
                index_list.append(str(i) + '_' + '{:02d}'.format(j))
            index = pd.DataFrame(index_list, columns=['FORE_data'])

            obs_prophet_M = df.iloc[-foretimes:, :9]
            obs_prophet_M = obs_prophet_M.reset_index(drop=True)

            # seq2seq
            anen_indices = [3, 6, 4, 7, 5, 8]
            pred_input_tensor = torch.from_numpy(
                float_data[-delay - seqence_len:-delay][np.newaxis, :].astype(np.float32)).to(device)
            pred_decoder_input_tensor = torch.from_numpy(
                float_data[-seqence_len:][np.newaxis, :][:, :, 3:13].astype(np.float32)).to(device)
            pred_anen_feature_tensor = torch.from_numpy(
                float_data[-seqence_len:, anen_indices].astype(np.float32)[np.newaxis, :]).to(device)
            pred_t2m = t2m_predictor.predict(pred_input_tensor, pred_decoder_input_tensor,
                                             pred_anen_feature_tensor).cpu().numpy()[0][-delay:, :]
            pred_rh2m = rh2m_predictor.predict(pred_input_tensor, pred_decoder_input_tensor,
                                               pred_anen_feature_tensor).cpu().numpy()[0][-delay:, :]
            pred_w10m = w10m_predictor.predict(pred_input_tensor, pred_decoder_input_tensor,
                                               pred_anen_feature_tensor).cpu().numpy()[0][-delay:, :]
            pred = np.c_[pred_t2m[:, 0], pred_rh2m[:, 1], pred_w10m[:, 2]]
            fill_row = np.array([[0, 0, 0]])
            pred = scaler.inverse_transform(pred, ['t2m_obs', 'rh2m_obs', 'w10m_obs'])
            seq2seq_pred_submit = np.r_[fill_row, pred]
            seq2seq_pred_df = pd.DataFrame(seq2seq_pred_submit,
                                           columns=['t2m_seq2seq_global', 'rh2m_seq2seq_global', 'w10m_seq2seq_global'])
            seq2seq_pred_df.iloc[:4, :] = obs_prophet_M.iloc[:4, :3]

            whole_df = pd.concat([index, obs_prophet_M, seq2seq_pred_df], axis=1)
            if submit_csv is None:
                submit_csv = whole_df
            else:
                submit_csv = pd.concat([submit_csv, whole_df], axis=0)
            # print(whole_df)
            print(str(i) + 'done!')
        submit_csv.to_csv(os.path.join(dst_dir,dst_date + '.csv'), index=False)
        df = pd.read_csv(os.path.join(dst_dir,dst_date + '.csv'), index_col=0)
        for i in range(len(df)):
            for col in df.columns:
                if np.isnan(df.iloc[i][col]):
                    df.iloc[i][col] = df.iloc[i][col.split('_')[0] + '_obs']
        df.to_csv(os.path.join(dst_dir,dst_date + '.csv'))

