import pandas as pd
from datetime import datetime
import os

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l
begin_date='2018-10-28'
end_date='2018-11-03'
dates=datelist(begin_date,end_date)
if not os.path.exists('obs'):
    os.mkdir('obs')
if not os.path.exists('fore'):
    os.mkdir('fore')

if __name__=='__main__':
    for date in dates:
        obs_and_M_filepath = 'obs_and_M/' + date + '.csv'
        obs_and_M = pd.read_csv(obs_and_M_filepath)
        print(obs_and_M.info())
        for col in obs_and_M.columns:
            obs_and_M[col] = obs_and_M[col].fillna(-9999)
        obs_and_M.round(3)
        obs_and_M['FORE_data'] = '  ' + obs_and_M['FORE_data']
        obs = pd.DataFrame(obs_and_M, columns=['FORE_data', 't2m_obs', 'rh2m_obs', 'w10m_obs'])
        obs.columns = ['  OBS_data', '       t2m', '      rh2m', '      w10m']

        obs.to_csv('obs/' + date + '_1_obs.csv', index=False, float_format='%.03f')

        M = pd.DataFrame(obs_and_M, columns=['FORE_data', 't2m_M', 'rh2m_M', 'w10m_M'])
        M.columns = ['FORE_data', '       t2m', '      rh2m', '      w10m']
        M.to_csv('fore/' + date + '_1_M.csv', index=False, float_format='%.03f')
