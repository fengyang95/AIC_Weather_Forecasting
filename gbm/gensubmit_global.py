import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from collections import OrderedDict
import json
import argparse
import os
# 加上48小时前obs 信息
# 处理 RAIN 值 去除 35以上数值
target_list=['t2m','rh2m','w10m']
from datetime import timedelta
from datetime import datetime
def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', dest='log_level', default='info', type=str,
                        help='Logging level.')
    parser.add_argument('--model_dir', dest='model_dir',
                        default='../checkpoints/lgb_global',
                        type=str)
    parser.add_argument('--data_dir',dest='data_dir',
                        default='../data/testb7/merge',type=str)

    parser.add_argument('--dst_dir',dest='dst_dir',
                        default='../result/lgb_global')
    parser.add_argument('--first_day',dest='first_day',
                        default='20181028',type=str)
    parser.add_argument('--last_day',dest='last_day',
                        default='20181103',type=str)


    opt = parser.parse_args()

    feature_columns = ['t2m_obs', 'rh2m_obs', 'w10m_obs', 'psur_obs', 'q2m_obs', 'u10m_obs',
                       'v10m_obs', 'RAIN_obs',
                       't2m_prophet', 'rh2m_prophet', 'w10m_prophet',
                       't2m_M', 'rh2m_M', 'w10m_M', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                       'psfc_M', 'q2m_M', 'u10m_M', 'v10m_M',
                       'SWD_M', 'GLW_M', 'HFX_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M',
                       'TC850_M', 'TC700_M', 'TC500_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M',
                       'location_90001', 'location_90002', 'location_90003', 'location_90004',
                       'location_90005', 'location_90006', 'location_90007', 'location_90008',
                       'location_90009', 'location_90010']
    if opt.model_dir.endswith('_q'):
        feature_columns = feature_columns + ['Q975_M', 'Q925_M', 'Q850_M', 'Q700_M', 'Q500_M', 'LH_M']

    history_num = 24

    begin_dates = datelist(pd.to_datetime(opt.first_day)-timedelta(days=2),
                           pd.to_datetime(opt.last_day)-timedelta(days=2))
    dst_dates = datelist(opt.first_day,opt.last_day)
    end_dates = datelist(pd.to_datetime(opt.first_day)+timedelta(days=1),pd.to_datetime(opt.last_day)+timedelta(days=1))

    model_dir =opt.model_dir
    data_dir = opt.data_dir
    dst_dir =opt.dst_dir
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for begin_date,dst_date,end_date in zip(begin_dates,dst_dates,end_dates):
        end_date=end_date+' 12-00-00'
        whole_submit = None
        for i in range(90001, 90011):
            index_list = []
            for j in range(37):
                index_list.append(str(i) + '_' + '{:02d}'.format(j))
            index = pd.DataFrame(index_list, columns=['FORE_data'])

            results = []
            for feature_index in range(3):

                lgb_model = joblib.load(os.path.join(model_dir,'model_feature_' + str(feature_index) + '.m'))


                data_pre = pd.read_csv(os.path.join(data_dir,'merged_' + str(i) + '.csv'), index_col=0)

                data_pre.index = pd.to_datetime(data_pre.index)
                data_pre['hour'] = data_pre.index.hour
                data_pre['month'] = data_pre.index.month
                hour_period = 24 / (2 * np.pi)
                data_pre['hour_cos'] = np.cos(data_pre.index.hour / hour_period)
                data_pre['hour_sin'] = np.sin(data_pre.index.hour / hour_period)

                month_period = 12 / (2 * np.pi)
                data_pre['month_cos'] = np.cos(data_pre.index.month / month_period)
                data_pre['month_sin'] = np.sin(data_pre.index.month / month_period)

                for j in range(90001, 90011):
                    data_pre['location_' + str(j)] = [0.] * len(data_pre)
                data_pre['location_' + str(i)] = [1.] * len(data_pre)

                data_pre['u10m_obs']=data_pre['u10m_obs']/data_pre['w10m_obs']
                data_pre['v10m_obs']=data_pre['v10m_obs']/data_pre['w10m_obs']
                data_pre['u10m_M']=data_pre['u10m_M']/data_pre['w10m_M']
                data_pre['v10m_M']=data_pre['v10m_M']/data_pre['w10m_M']

                data_pre = pd.DataFrame(data_pre, columns=feature_columns)
                data_pre = data_pre[begin_date:end_date]
                for col in data_pre.columns:
                    data_pre[col] = data_pre[col].fillna(data_pre[col].mean())

                pre_data_float = np.array(data_pre)
                # print(pre_data_float.shape)
                history = list(pre_data_float[24:48, feature_index])
                result = []
                for k in range(37):
                    row_data = pre_data_float[48 + k:48 + k + 1, 8:]
                    curr_history = np.array(history)[k:k + history_num]
                    obs_48h_ago = pre_data_float[k:k + 1, :8]
                    # print(row_data.shape)
                    # print(curr_history.shape)
                    curr_history = curr_history.reshape((1, -1))
                    X = np.c_[row_data, curr_history, obs_48h_ago]
                    #print(X.shape)
                    y = lgb_model.predict(X)
                    # print(y.shape)
                    if k < 4:
                        result.append(pre_data_float[48 + k, feature_index])
                        history.append(pre_data_float[48 + k, feature_index])
                    else:
                        result.append(y[0])
                        history.append(y[0])
                # print('result:', result)
                rmse = np.sqrt(mean_squared_error(pre_data_float[48:, feature_index], np.array(result)))
                rmse_M = np.sqrt(
                    mean_squared_error(pre_data_float[48:, feature_index], pre_data_float[48:, feature_index +11]))
                results.append(result)

                print('rmse:', rmse)
                print('rmse_M:', rmse_M)
                print('score:', (rmse_M - rmse) / rmse_M)
            suffix=opt.model_dir.split('/')[-1]
            submit = pd.DataFrame(np.array(results).T, columns=['t2m_'+suffix, 'rh2m_'+suffix, 'w10m_'+suffix])
            submit = pd.concat([index, submit], axis=1)
            if whole_submit is None:
                whole_submit = submit
            else:
                whole_submit = pd.concat([whole_submit, submit], axis=0)

        whole_submit.to_csv(os.path.join(dst_dir,dst_date + '.csv'), index=False)



















