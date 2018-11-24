import pandas as pd
import numpy as np
import os
from datetime import datetime

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

if not os.path.exists('merged'):
    os.mkdir('merged')
if __name__=='__main__':
    dates=datelist('20181028','20181103')
    for date in dates:
        obs_and_M_results = pd.read_csv('obs_and_M/' + date + '.csv')
        catboost_results=pd.read_csv('catboost/'+date+'.csv')
        catboost_q_resultls=pd.read_csv('catboost_q/'+date+'.csv')
        lgb_results = pd.read_csv('lgb/' + date + '.csv')
        lgb_q_results = pd.read_csv('lgb_q/' + date + '.csv')

        lgb_global_results = pd.read_csv('lgb_global/' + date + '.csv')
        lgb_global_q_results=pd.read_csv('lgb_global_q/'+date+'.csv')
        catboost_global_results=pd.read_csv('catboost_global/'+date+'.csv')
        catboost_global_q_results=pd.read_csv('catboost_global_q/'+date+'.csv')

        seq2seq_feature48_results = pd.read_csv('seq2seq_feature48/' + date + '.csv')

        FORE_data = pd.DataFrame(obs_and_M_results, columns=['FORE_data'])
        obs_and_M_results = pd.DataFrame(obs_and_M_results, columns=['t2m_obs', 'rh2m_obs', 'w10m_obs',
                                                                    't2m_M', 'rh2m_M', 'w10m_M'])

        catboost_results=pd.DataFrame(catboost_results,columns=['t2m_catboost','rh2m_catboost','w10m_catboost'])
        catboost_q_resultls=pd.DataFrame(catboost_q_resultls,columns=['t2m_catboost_q','rh2m_catboost_q','w10m_catboost_q'])

        lgb_results = pd.DataFrame(lgb_results, columns=['t2m_lgb', 'rh2m_lgb', 'w10m_lgb'])
        lgb_q_results = pd.DataFrame(lgb_q_results, columns=['t2m_lgb_q', 'rh2m_lgb_q', 'w10m_lgb_q'])

        lgb_global_results = pd.DataFrame(lgb_global_results,
                                          columns=['t2m_lgb_global', 'rh2m_lgb_global', 'w10m_lgb_global'])

        lgb_global_q_results=pd.DataFrame(lgb_global_q_results,
                                          columns=['t2m_lgb_global_q','rh2m_lgb_global_q',
                                                   'w10m_lgb_global_q'])
        catboost_global_results = pd.DataFrame(catboost_global_results,
                                          columns=['t2m_catboost_global', 'rh2m_catboost_global', 'w10m_catboost_global'])

        catboost_global_q_results = pd.DataFrame(catboost_global_q_results,
                                            columns=['t2m_catboost_global_q', 'rh2m_catboost_global_q',
                                                     'w10m_catboost_global_q'])

        seq2seq_feature48_results = pd.DataFrame(seq2seq_feature48_results, columns=[
            't2m_seq2seq_global', 'rh2m_seq2seq_global', 'w10m_seq2seq_global'
        ])


        merged = pd.concat(
            [FORE_data, obs_and_M_results,catboost_results,catboost_q_resultls,lgb_results,lgb_q_results,
             lgb_global_results,lgb_global_q_results,catboost_global_results,catboost_global_q_results,
             seq2seq_feature48_results], axis=1)

        for i in range(len(merged)):
            if i % 37 < 4:
                for col in merged.columns:
                    if col in ['FORE_data', 't2m_M', 'rh2m_M', 'w10m_M']:
                        continue
                    else:
                        # 错误修改方法
                        # merged.loc[i][col]=merged.loc[i][col.split('_')[0]+'_obs']
                        if not np.isnan(merged.loc[i, col.split('_')[0] + '_obs']):
                            merged.loc[i, col] =merged.loc[i, col.split('_')[0] + '_obs']
        merged.to_csv('merged/' + date + '.csv', index=False)

