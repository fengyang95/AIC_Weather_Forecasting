import pandas as pd
from datetime import datetime

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

if __name__=='__main__':
    dates = datelist('20181028', '20181103')
    for date in dates:
        file_path = '../' + date + '.csv'
        data = pd.read_csv(file_path)

        t2m_merge5 = (data['t2m_catboost'] + data['t2m_catboost_q'] + data['t2m_lgb'] + data['t2m_lgb_q']
                      + data['t2m_seq2seq_global']) / 5
        rh2m_merge5 = (data['rh2m_catboost'] + data['rh2m_catboost_q'] + data['rh2m_lgb'] + data['rh2m_lgb_q'] +
                       data['rh2m_seq2seq_global']) / 5
        w10m_merge5 = (data['w10m_catboost'] + data['w10m_catboost_q'] + data['w10m_lgb'] + data[
            'w10m_lgb_q']+data['w10m_seq2seq_global']) / 5

        data['t2m_v1'] = t2m_merge5
        data['rh2m_v1'] = rh2m_merge5
        data['w10m_v1'] = w10m_merge5

        t2m_merge2 = (data['t2m_catboost'] +data['t2m_lgb']) / 2
        rh2m_merge2 = (data['rh2m_catboost'] + data['rh2m_lgb']) / 2
        w10m_merge2 = (data['w10m_catboost'] + data['w10m_lgb']) / 2

        data['t2m_v2'] = t2m_merge2
        data['rh2m_v2'] = rh2m_merge2
        data['w10m_v2'] = w10m_merge2

        t2m_merge6 = (data['t2m_catboost'] + data['t2m_catboost_q'] + data['t2m_lgb'] + data['t2m_lgb_q']
                      + 2 * data['t2m_seq2seq_global']) / 6
        rh2m_merge6 = (data['rh2m_catboost'] + data['rh2m_catboost_q'] + data['rh2m_lgb'] + data['rh2m_lgb_q'] +
                       2 * data['rh2m_seq2seq_global']) / 6
        w10m_merge6 = (data['w10m_catboost'] + data['w10m_catboost_q'] + data['w10m_lgb'] + data[
            'w10m_lgb_q'] + 2 * data['w10m_seq2seq_global']) / 6

        data['t2m_v3'] = t2m_merge6
        data['rh2m_v3'] = rh2m_merge6
        data['w10m_v3'] = w10m_merge6

        t2m_merge3=(data['t2m_lgb']+data['t2m_catboost']+data['t2m_seq2seq_global'])/3
        rh2m_merge3=(data['rh2m_lgb']+data['rh2m_catboost']+data['rh2m_seq2seq_global'])/3
        w10m_merge3=(data['w10m_lgb']+data['w10m_catboost']+data['w10m_seq2seq_global'])/3
        data['t2m_v4']=t2m_merge3
        data['rh2m_v4']=rh2m_merge3
        data['w10m_v4']=w10m_merge3

        submission_v1 = pd.DataFrame(data, columns=['FORE_data', 't2m_v1', 'rh2m_v1', 'w10m_v1'])
        submission_v1.columns = ['FORE_data', '       t2m', '      rh2m', '      w10m']
        submission_v1.to_csv(date + '_v1.csv', index=False)

        submission_v2 = pd.DataFrame(data, columns=['FORE_data', 't2m_v2', 'rh2m_v2', 'w10m_v2'])
        submission_v2.columns = ['FORE_data', '       t2m', '      rh2m', '      w10m']
        submission_v2.to_csv(date + '_v2.csv', index=False)

        submission_v3 = pd.DataFrame(data, columns=['FORE_data', 't2m_v3', 'rh2m_v3', 'w10m_v3'])
        submission_v3.columns = ['FORE_data', '       t2m', '      rh2m', '      w10m']
        submission_v3.to_csv(date + '_v3.csv', index=False)

        submission_v4 = pd.DataFrame(data, columns=['FORE_data', 't2m_v4', 'rh2m_v4', 'w10m_v4'])
        submission_v4.columns = ['FORE_data', '       t2m', '      rh2m', '      w10m']
        submission_v4.to_csv(date + '_v4.csv', index=False)

