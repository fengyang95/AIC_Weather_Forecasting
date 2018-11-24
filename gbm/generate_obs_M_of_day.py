import pandas as pd
import argparse
import logging
from datetime import datetime
import os

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

parser = argparse.ArgumentParser()
parser.add_argument('--log-level', dest='log_level', default='info', type=str,
                        help='Logging level.')
parser.add_argument('--src_dir', dest='src_dir', type=str, default='../data/testb7/obs_and_M', help='src csv file dir')
parser.add_argument('--str_begin_date',dest='begin_date',type=str,default='20181028',help='begin date')
parser.add_argument('--str_end_date',dest='end_date',type=str,default='20181103',help='end date')
opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if __name__=='__main__':

    dst_dir='../result/obs_and_M'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    dst_dates = datelist(opt.begin_date, opt.end_date)
    for dst_date in dst_dates:
        whole = None
        date=str(dst_date)
        for i in range(90001, 90011):
            file_path = os.path.join(opt.src_dir,str(i) + '.csv')
            datai = pd.read_csv(file_path)
            str_date=date[:4]+date[5:7]+date[8:10]
            target_data = datai[datai['Time'].str.contains(str_date)]
            index_list = []
            for j in range(37):
                index_list.append(str(i) + '_' + '{:02d}'.format(j))
            target_data = pd.DataFrame(data=target_data,
                                       columns=['t2m_obs', 'rh2m_obs', 'w10m_obs', 't2m_M', 'rh2m_M', 'w10m_M'])
            target_data.reset_index(inplace=True)
            index = pd.DataFrame(index_list, columns=['FORE_data'])
            target_data = pd.concat([index, target_data], axis=1)
            target_data = target_data.drop(['index'], axis=1)

            if target_data is None:
                whole = target_data
            else:
                whole = pd.concat([whole, target_data], axis=0)
        whole.to_csv(os.path.join(dst_dir,date + '.csv'), index=False)
        print(dst_date,' done!')


