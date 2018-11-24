import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import json
import os
from datetime import datetime
from collections import OrderedDict
def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l
if not os.path.exists('score'):
    os.mkdir('score')

if __name__=='__main__':
    dates=datelist('20181028','20181103')
    for date in dates:
        file_path = os.path.join('merged',date + '.csv')
        data = pd.read_csv(file_path)
        rows_drop = [i for i,_ in enumerate(data['FORE_data']) if data['FORE_data'][i].split('_')[-1] in ['00','01','02','03']]
        data=data.drop(rows_drop,axis=0)

        data.dropna(axis=0, how='any', inplace=True)
        if len(data) ==0:
            break
        t2m_merge2=(data['t2m_lgb']+data['t2m_catboost'])/2
        t2m_merge3=(data['t2m_lgb']+data['t2m_catboost']+data['t2m_seq2seq_global'])/3
        t2m_merge5=(data['t2m_lgb']+data['t2m_lgb_q']+data['t2m_catboost']+data['t2m_catboost_q']+data['t2m_seq2seq_global'])/5
        t2m_merge3_global=(data['t2m_lgb_global']+data['t2m_catboost_global']+data['t2m_seq2seq_global'])/3
        t2m_merge5_global=(data['t2m_lgb_global']+data['t2m_lgb_global_q']+data['t2m_catboost_global']+
                           data['t2m_catboost_global_q']+data['t2m_seq2seq_global'])/5

        rh2m_merge2=(data['rh2m_lgb']+data['rh2m_catboost'])/2
        rh2m_merge3 = (data['rh2m_lgb'] + data['rh2m_catboost'] + data['rh2m_seq2seq_global']) / 3
        rh2m_merge5 = (data['rh2m_lgb'] + data['rh2m_lgb_q'] + data['rh2m_catboost'] + data['rh2m_catboost_q'] +
                       data['rh2m_seq2seq_global']) / 5
        rh2m_merge3_global = (data['rh2m_lgb_global'] + data['rh2m_catboost_global'] + data['rh2m_seq2seq_global']) / 3
        rh2m_merge5_global = (data['rh2m_lgb_global'] + data['rh2m_lgb_global_q'] + data['rh2m_catboost_global'] +
                             data['rh2m_catboost_global_q'] + data['rh2m_seq2seq_global']) / 5

        w10m_merge2=(data['w10m_lgb']+data['w10m_catboost'])/2
        w10m_merge3 = (data['w10m_lgb'] + data['w10m_catboost'] + data['w10m_seq2seq_global']) / 3
        w10m_merge5 = (data['w10m_lgb'] + data['w10m_lgb_q'] + data['w10m_catboost'] + data['w10m_catboost_q'] + data[
            'w10m_seq2seq_global']) / 5
        w10m_merge3_global = (data['w10m_lgb_global'] + data['w10m_catboost_global'] + data['w10m_seq2seq_global']) / 3
        w10m_merge5_global = (data['w10m_lgb_global'] + data['w10m_lgb_global_q'] + data['w10m_catboost_global'] +
                             data['w10m_catboost_global_q'] + data['w10m_seq2seq_global']) / 5


        rmse_t2m_M = np.sqrt(mean_squared_error(data['t2m_obs'], data['t2m_M']))
        rmse_t2m_lgb=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_lgb']))
        rmse_t2m_lgb_q=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_lgb_q']))
        rmse_t2m_catboost=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_catboost']))
        rmse_t2m_catboost_q=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_catboost_q']))
        rmse_t2m_lgb_global=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_lgb_global']))
        rmse_t2m_lgb_global_q=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_lgb_global_q']))
        rmse_t2m_catboost_global=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_catboost_global']))
        rmse_t2m_catboost_global_q=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_catboost_global_q']))
        rmse_t2m_seq2seq_global=np.sqrt(mean_squared_error(data['t2m_obs'],data['t2m_seq2seq_global']))
        rmse_t2m_merge2=np.sqrt(mean_squared_error(data['t2m_obs'],t2m_merge2))
        rmse_t2m_merge3=np.sqrt(mean_squared_error(data['t2m_obs'],t2m_merge3))
        rmse_t2m_merge3_global=np.sqrt(mean_squared_error(data['t2m_obs'],t2m_merge3_global))
        rmse_t2m_merge5=np.sqrt(mean_squared_error(data['t2m_obs'],t2m_merge5))
        rmse_t2m_merge5_global=np.sqrt(mean_squared_error(data['t2m_obs'],t2m_merge5_global))

        rmse_rh2m_M = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_M']))
        rmse_rh2m_lgb = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_lgb']))
        rmse_rh2m_lgb_q = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_lgb_q']))
        rmse_rh2m_catboost = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_catboost']))
        rmse_rh2m_catboost_q = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_catboost_q']))
        rmse_rh2m_lgb_global = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_lgb_global']))
        rmse_rh2m_lgb_global_q = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_lgb_global_q']))
        rmse_rh2m_catboost_global = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_catboost_global']))
        rmse_rh2m_catboost_global_q = np.sqrt(mean_squared_error(data['rh2m_obs'], data['rh2m_catboost_global_q']))
        rmse_rh2m_seq2seq_global=np.sqrt(mean_squared_error(data['rh2m_obs'],data['rh2m_seq2seq_global']))
        rmse_rh2m_merge2=np.sqrt(mean_squared_error(data['rh2m_obs'],rh2m_merge2))
        rmse_rh2m_merge3 = np.sqrt(mean_squared_error(data['rh2m_obs'], rh2m_merge3))
        rmse_rh2m_merge3_global = np.sqrt(mean_squared_error(data['rh2m_obs'], rh2m_merge3_global))
        rmse_rh2m_merge5 = np.sqrt(mean_squared_error(data['rh2m_obs'], rh2m_merge5))
        rmse_rh2m_merge5_global = np.sqrt(mean_squared_error(data['rh2m_obs'], rh2m_merge5_global))

        rmse_w10m_M = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_M']))
        rmse_w10m_lgb = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_lgb']))
        rmse_w10m_lgb_q = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_lgb_q']))
        rmse_w10m_catboost = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_catboost']))
        rmse_w10m_catboost_q = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_catboost_q']))
        rmse_w10m_lgb_global = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_lgb_global']))
        rmse_w10m_lgb_global_q = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_lgb_global_q']))
        rmse_w10m_catboost_global = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_catboost_global']))
        rmse_w10m_catboost_global_q = np.sqrt(mean_squared_error(data['w10m_obs'], data['w10m_catboost_global_q']))
        rmse_w10m_seq2seq_global=np.sqrt(mean_squared_error(data['w10m_obs'],data['w10m_seq2seq_global']))
        rmse_w10m_merge2=np.sqrt(mean_squared_error(data['w10m_obs'],w10m_merge2))
        rmse_w10m_merge3 = np.sqrt(mean_squared_error(data['w10m_obs'], w10m_merge3))
        rmse_w10m_merge3_global = np.sqrt(mean_squared_error(data['w10m_obs'], w10m_merge3_global))
        rmse_w10m_merge5 = np.sqrt(mean_squared_error(data['w10m_obs'], w10m_merge5))
        rmse_w10m_merge5_global = np.sqrt(mean_squared_error(data['w10m_obs'], w10m_merge5_global))

        score_t2m_lgb=(rmse_t2m_M-rmse_t2m_lgb)/rmse_t2m_M
        score_t2m_lgb_q=(rmse_t2m_M-rmse_t2m_lgb_q)/rmse_t2m_M
        score_t2m_catboost=(rmse_t2m_M-rmse_t2m_catboost)/rmse_t2m_M
        score_t2m_catboost_q=(rmse_t2m_M-rmse_t2m_catboost_q)/rmse_t2m_M
        score_t2m_lgb_global=(rmse_t2m_M-rmse_t2m_lgb_global)/rmse_t2m_M
        score_t2m_lgb_global_q=(rmse_t2m_M-rmse_t2m_lgb_global_q)/rmse_t2m_M
        score_t2m_catboost_global=(rmse_t2m_M-rmse_t2m_catboost_global)/rmse_t2m_M
        score_t2m_catboost_global_q=(rmse_t2m_M-rmse_t2m_catboost_global_q)/rmse_t2m_M
        score_t2m_seq2seq_global=(rmse_t2m_M-rmse_t2m_seq2seq_global)/rmse_t2m_M
        score_t2m_merge2=(rmse_t2m_M-rmse_t2m_merge2)/rmse_t2m_M
        score_t2m_merge3=(rmse_t2m_M-rmse_t2m_merge3)/rmse_t2m_M
        score_t2m_merge3_global=(rmse_t2m_M-rmse_t2m_merge3_global)/rmse_t2m_M
        score_t2m_merge5=(rmse_t2m_M-rmse_t2m_merge5)/rmse_t2m_M
        score_t2m_merge5_global=(rmse_t2m_M-rmse_t2m_merge5_global)/rmse_t2m_M

        score_rh2m_lgb = (rmse_rh2m_M - rmse_rh2m_lgb) / rmse_rh2m_M
        score_rh2m_lgb_q = (rmse_rh2m_M - rmse_rh2m_lgb_q) / rmse_rh2m_M
        score_rh2m_catboost = (rmse_rh2m_M - rmse_rh2m_catboost) / rmse_rh2m_M
        score_rh2m_catboost_q = (rmse_rh2m_M - rmse_rh2m_catboost_q) / rmse_rh2m_M
        score_rh2m_lgb_global = (rmse_rh2m_M - rmse_rh2m_lgb_global) / rmse_rh2m_M
        score_rh2m_lgb_global_q = (rmse_rh2m_M - rmse_rh2m_lgb_global_q) / rmse_rh2m_M
        score_rh2m_catboost_global = (rmse_rh2m_M - rmse_rh2m_catboost_global) / rmse_rh2m_M
        score_rh2m_catboost_global_q = (rmse_rh2m_M - rmse_rh2m_catboost_global_q) / rmse_rh2m_M
        score_rh2m_seq2seq_global = (rmse_rh2m_M - rmse_rh2m_seq2seq_global) / rmse_rh2m_M
        score_rh2m_merge2=(rmse_rh2m_M-rmse_rh2m_merge2)/rmse_rh2m_M
        score_rh2m_merge3 = (rmse_rh2m_M - rmse_rh2m_merge3) / rmse_rh2m_M
        score_rh2m_merge3_global = (rmse_rh2m_M - rmse_rh2m_merge3_global) / rmse_rh2m_M
        score_rh2m_merge5 = (rmse_rh2m_M - rmse_rh2m_merge5) / rmse_rh2m_M
        score_rh2m_merge5_global = (rmse_rh2m_M - rmse_rh2m_merge5_global) / rmse_rh2m_M

        score_w10m_lgb = (rmse_w10m_M - rmse_w10m_lgb) / rmse_w10m_M
        score_w10m_lgb_q = (rmse_w10m_M - rmse_w10m_lgb_q) / rmse_w10m_M
        score_w10m_catboost = (rmse_w10m_M - rmse_w10m_catboost) / rmse_w10m_M
        score_w10m_catboost_q = (rmse_w10m_M - rmse_w10m_catboost_q) / rmse_w10m_M
        score_w10m_lgb_global = (rmse_w10m_M - rmse_w10m_lgb_global) / rmse_w10m_M
        score_w10m_lgb_global_q = (rmse_w10m_M - rmse_w10m_lgb_global_q) / rmse_w10m_M
        score_w10m_catboost_global = (rmse_w10m_M - rmse_w10m_catboost_global) / rmse_w10m_M
        score_w10m_catboost_global_q = (rmse_w10m_M - rmse_w10m_catboost_global_q) / rmse_w10m_M
        score_w10m_seq2seq_global = (rmse_w10m_M - rmse_w10m_seq2seq_global) / rmse_w10m_M
        score_w10m_merge2=(rmse_w10m_M-rmse_w10m_merge2)/rmse_w10m_M
        score_w10m_merge3 = (rmse_w10m_M - rmse_w10m_merge3) / rmse_w10m_M
        score_w10m_merge3_global = (rmse_w10m_M - rmse_w10m_merge3_global) / rmse_w10m_M
        score_w10m_merge5 = (rmse_w10m_M - rmse_w10m_merge5) / rmse_w10m_M
        score_w10m_merge5_global = (rmse_w10m_M - rmse_w10m_merge5_global) / rmse_w10m_M

        score_lgb=(score_t2m_lgb+score_rh2m_lgb+score_w10m_lgb)/3
        score_lgb_q=(score_t2m_lgb_q+score_rh2m_lgb_q+score_w10m_lgb_q)/3
        score_catboost=(score_t2m_catboost+score_rh2m_catboost+score_w10m_catboost)/3
        score_catboost_q=(score_t2m_catboost_q+score_rh2m_catboost_q+score_w10m_catboost_q)/3
        score_lgb_global=(score_t2m_lgb_global+score_rh2m_lgb_global+score_w10m_lgb_global)/3
        score_lgb_global_q=(score_t2m_lgb_global_q+score_rh2m_lgb_global_q+score_w10m_lgb_global_q)/3
        score_catboost_global=(score_t2m_catboost_global+score_rh2m_catboost_global+score_w10m_catboost_global)/3
        score_catboost_global_q=(score_t2m_catboost_global_q+score_rh2m_catboost_global_q+score_w10m_catboost_global_q)/3
        score_seq2seq_global=(score_t2m_seq2seq_global+score_rh2m_seq2seq_global+score_w10m_seq2seq_global)/3
        score_merge2=(score_t2m_merge2+score_rh2m_merge2+score_w10m_merge2)/3
        score_merge3=(score_t2m_merge3+score_rh2m_merge3+score_w10m_merge3)/3
        score_merge3_global=(score_t2m_merge3_global+score_rh2m_merge3_global+score_w10m_merge3_global)/3
        score_merge5=(score_t2m_merge5+score_rh2m_merge5+score_w10m_merge5)/3
        score_merge5_global=(score_t2m_merge5_global+score_rh2m_merge5_global+score_w10m_merge5_global)/3

        result_dict = {
            't2m score': {
                'lgb':score_t2m_lgb,
                'lgb_q':score_t2m_lgb_q,
                'catboost':score_t2m_catboost,
                'catboost_q':score_t2m_catboost_q,
                'lgb_global':score_t2m_lgb_global,
                'lgb_global_q':score_t2m_lgb_global_q,
                'catboost_global':score_t2m_catboost_global,
                'catboost_global_q': score_t2m_catboost_global_q,
                'seq2seq_global':score_t2m_seq2seq_global,
                'merge2':score_t2m_merge2,
                'merge3':score_t2m_merge3,
                'merge3_global':score_t2m_merge3_global,
                'merge5':score_t2m_merge5,
                'merge5_global':score_t2m_merge5_global,
            },
            'rh2m_score': {
                'lgb': score_rh2m_lgb,
                'lgb_q': score_rh2m_lgb_q,
                'catboost': score_rh2m_catboost,
                'catboost_q': score_rh2m_catboost_q,
                'lgb_global': score_rh2m_lgb_global,
                'lgb_global_q': score_rh2m_lgb_global_q,
                'catboost_global': score_rh2m_catboost_global,
                'catboost_global_q': score_rh2m_catboost_global_q,
                'seq2seq_global': score_rh2m_seq2seq_global,
                'merge2':score_rh2m_merge2,
                'merge3':score_rh2m_merge3,
                'merge3_global':score_rh2m_merge3_global,
                'merge5':score_rh2m_merge5,
                'merge5_global':score_rh2m_merge5_global
            },
            'w10m_score': {
                'lgb': score_w10m_lgb,
                'lgb_q': score_w10m_lgb_q,
                'catboost': score_w10m_catboost,
                'catboost_q': score_w10m_catboost_q,
                'lgb_global': score_w10m_lgb_global,
                'lgb_global_q': score_w10m_lgb_global_q,
                'catboost_global': score_w10m_catboost_global,
                'catboost_global_q': score_w10m_catboost_global_q,
                'seq2seq_global': score_w10m_seq2seq_global,
                'merge2':score_w10m_merge2,
                'merge3':score_w10m_merge3,
                'merge3_global':score_w10m_merge3_global,
                'merge5':score_w10m_merge5,
                'merge5_global':score_w10m_merge5_global
            },
            'score': {
                'lgb': score_lgb,
                'lgb_q': score_lgb_q,
                'catboost': score_catboost,
                'catboost_q': score_catboost_q,
                'lgb_global': score_lgb_global,
                'lgb_global_q': score_lgb_global_q,
                'catboost_global': score_catboost_global,
                'catboost_global_q': score_catboost_global_q,
                'seq2seq_global': score_seq2seq_global,
                'merge2':score_merge2,
                'merge3':score_merge3,
                'merge3_global':score_merge3_global,
                'merge5':score_merge5,
                'merge5_global':score_merge5_global,
            }
        }
        for obs in result_dict.keys():
            result_dict[obs] = OrderedDict(
                (k, v) for k, v in sorted(result_dict[obs].items(), key=lambda x: x[1], reverse=True))
        json_content = json.dumps(result_dict)
        f = open(os.path.join('score','score_' + date + '.json'), 'w')
        f.write(json_content)
        f.close()


