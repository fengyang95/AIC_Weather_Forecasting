import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

dates=datelist('20181028','20181103')

vis_dir='vis'
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

if __name__=='__main__':
    for date in dates:
        df = pd.read_csv('merged/'+date + '.csv')
        df['t2m_obs'].plot(label='obs')
        df['t2m_seq2seq_global'].plot(label='seq2seq')
        df['t2m_catboost'].plot(label='catboost')
        df['t2m_lgb'].plot(label='lgb')

        plt.legend()
        plt.title(date+'t2m')
        plt.savefig(os.path.join(vis_dir,date + '_t2m.jpg'))
        plt.show()

        df['rh2m_obs'].plot(label='obs')
        df['rh2m_seq2seq_global'].plot(label='seq2seq')
        df['rh2m_catboost_q'].plot(label='catboost')
        df['rh2m_lgb'].plot(label='lgb')

        plt.legend()
        plt.title(date+'rh2m')
        plt.savefig(os.path.join(vis_dir,date + '_rh2m.jpg'))
        plt.show()

        df['w10m_obs'].plot(label='obs')
        df['w10m_seq2seq_global'].plot(label='seq2seq')
        df['w10m_catboost'].plot(label='catboost')
        df['w10m_lgb'].plot(label='lgb')
        plt.legend()
        plt.title(date+'w10m')
        plt.savefig(os.path.join(vis_dir,date + '_w10m.jpg'))
        plt.show()

