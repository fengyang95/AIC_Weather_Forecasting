import argparse
import os
import logging
from netCDF4 import Dataset
import numpy as np
import pandas as pd

def nc2csv_obs_and_M(src_file_path, dst_dir):
    with Dataset(src_file_path) as nc:
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        stations = nc.variables['station'][:]
        date = nc.variables['date'][:]
        id_list = []
        for i in range(len(date)):
            for j in range(37):
                id = str(date[i])[:8] + '_' + '{:02d}'.format(j)
                id_list.append(id)
        ID = pd.Series(data=id_list, name='Time')
        for i in range(len(stations)):
            csv = pd.concat([ID], axis=1)
            for var in ['t2m_obs', 'rh2m_obs', 'w10m_obs', 't2m_M', 'rh2m_M', 'w10m_M']:
                var_arr = np.array(nc.variables[var][:])
                var_arr = np.squeeze(var_arr[:, :, i].reshape(-1, 1))
                var_arr[var_arr < -8000] = np.NaN
                csv[var] = var_arr
            csv.to_csv(os.path.join(dst_dir,str(stations[i]) + '.csv'), index=False)
            print(stations[i],' done!')



def nc2csv_merge_pre_and_next(src_file_path,str_lastday,dst_dir):
    with Dataset(src_file_path) as nc:
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        stations = nc.variables['station'][:]
        date = nc.variables['date'][:]
        id_list = []
        for i in range(len(date)):
            for j in range(24):
                id = str(date[i])[:8] + ' ' + '{:02d}'.format(j)
                id_list.append(id)
        for j in range(13):
            id = str_lastday + ' ' + '{:02d}'.format(j)
            id_list.append(id)
        Time = pd.to_datetime(id_list)
        ID = pd.Series(data=Time, name='Time')

        for i in range(len(stations)):
            csv = pd.concat([ID], axis=1)
            for var in nc.variables:
                if var.endswith('obs'):
                    var_arr = np.array(nc.variables[var][:])
                elif var.endswith('M'):
                    var_arr = np.array(nc.variables[var][:])
                    for j in range(1, var_arr.shape[0]):
                        for k in range(13):
                            pre = var_arr[j - 1, 24 + k, i]
                            current = var_arr[j, k, i]
                            if current == -9999:
                                var_arr[j, k, i] = pre
                            elif current != -9999 and pre != -9999:
                                var_arr[j, k, i] = (pre + current) / 2
                else:
                    continue
                var_arr_first = np.squeeze(var_arr[:, :24, i].reshape(-1, 1))
                var_arr_last = np.squeeze((var_arr[-1, 24:, i]).reshape(-1, 1))
                var_arr = np.r_[var_arr_first, var_arr_last]
                var_arr[var_arr < -8000] = np.NaN
                csv[var] = var_arr
            csv.to_csv(os.path.join(dst_dir,str(stations[i]) + '.csv'), index=False)
            print(stations[i],' done!')


def nc2csv_remain_all_info(src_file_path,str_lastday,dst_dir):
    with Dataset(src_file_path) as nc:
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        stations = nc.variables['station'][:]
        date = nc.variables['date'][:]
        id_list = []
        for i in range(len(date)):
            for j in range(24):
                id = str(date[i])[:8] + ' ' + '{:02d}'.format(j)
                id_list.append(id)
        for j in range(13):
            id = str_lastday + ' ' + '{:02d}'.format(j)
            id_list.append(id)
        Time = pd.to_datetime(id_list)
        ID = pd.Series(data=Time, name='Time')
        for i in range(len(stations)):
            csv = pd.concat([ID], axis=1)
            for var in nc.variables:
                var_arr_pre = None
                if var.endswith('obs'):
                    var_arr = np.array(nc.variables[var][:])
                elif var.endswith('M'):
                    var_arr = np.array(nc.variables[var][:])
                    var_arr_pre = np.array(nc.variables[var][:])
                    for j in range(1, var_arr.shape[0]):
                        for k in range(13):
                            pre = var_arr[j - 1, 24 + k, i]
                            var_arr_pre[j, k, i] = pre
                else:
                    continue
                if var_arr_pre is not None:
                    pre_first = np.squeeze(var_arr_pre[:, :24, i].reshape(-1, 1))
                    pre_last = np.squeeze((var_arr_pre[-1, 24:, i]).reshape(-1, 1))
                    var_arr_pre = np.r_[pre_first, pre_last]
                    var_arr_pre[var_arr_pre < -8000] = np.NaN
                    csv[var + '_pre'] = var_arr_pre
                var_arr_first = np.squeeze(var_arr[:, :24, i].reshape(-1, 1))
                var_arr_last = np.squeeze((var_arr[-1, 24:, i]).reshape(-1, 1))
                var_arr = np.r_[var_arr_first, var_arr_last]
                var_arr[var_arr < -8000] = np.NaN
                csv[var] = var_arr
            csv.to_csv(os.path.join(dst_dir, str(stations[i]) + '.csv'), index=False)
            print(stations[i],' done!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', dest='log_level',default='info',type=str,
                        help='Logging level.')
    parser.add_argument('--src_file_path',dest='src_file_path',default='../data/train/ai_challenger_wf2018_trainingset_20150301-20180531.nc',
                        type=str,help='Path to source nc file')
    parser.add_argument('--str_lastday',dest='str_lastday',default='20180601',type=str,help='last day in nc file')
    parser.add_argument('--dst_dir',dest='dst_dir',type=str,default='../data/train/remain_all_info',help='dst csv file dir')
    parser.add_argument('--method',dest='method',type=str,default='remain_all',help='method type must be remain_all or merge or origin')
    opt = parser.parse_args()
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)
    if opt.method=='remain_all':
        nc2csv_remain_all_info(src_file_path=opt.src_file_path,str_lastday=opt.str_lastday,dst_dir=opt.dst_dir)
    elif opt.method=='merge':
        nc2csv_merge_pre_and_next(src_file_path=opt.src_file_path,str_lastday=opt.str_lastday,dst_dir=opt.dst_dir)
    elif opt.method=='obs_and_M':
        nc2csv_obs_and_M(src_file_path=opt.src_file_path, dst_dir=opt.dst_dir)
