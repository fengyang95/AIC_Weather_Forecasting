import argparse
import logging
import pandas as pd
import numpy as np
import os

'''
策略 对于M值，首先考虑用前后两天的均值替代 然后考虑用相应obs值替代 最后考虑用均值替代
对于 obs值，首先考虑用前后两天均值替代，然后考虑用相应M值替代 最后考虑用均值替代
'''
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', dest='log_level',default='info',type=str,
                        help='Logging level.')
    parser.add_argument('--src_dir',dest='src_dir',type=str,default='../data/testb7/merge',help='src csv file dir')
    opt = parser.parse_args()
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)
    for j in range(90001,90011):
        file_name=os.path.join(opt.src_dir,str(j)+'.csv')
        data=pd.read_csv(file_name,index_col=0)
        length = len(data)

        data['psur_M'] = data['psfc_M']
        data.drop(['psfc_M'], axis=1, inplace=True)
        same_attr_list = ['t2m', 'rh2m', 'w10m', 'psur', 'q2m', 'd10m', 'u10m',
                          'v10m', 'RAIN']

        for i in range(1, length - 1):
            for attr in data.columns:
                if np.isnan(data.iloc[i][attr]):
                    if not np.isnan(data.iloc[i - 1][attr]) and not np.isnan(data.iloc[i + 1][attr]):
                        data.iloc[i][attr] = 0.5 * (data.iloc[i - 1][attr] + data.iloc[i + 1][attr])
                    elif attr.split('_')[0] in same_attr_list:
                        if attr.endswith('M'):
                            if not np.isnan(data.iloc[i][attr.split('_')[0]+'_obs']):
                                data.iloc[i][attr]=data.iloc[i][attr.split('_')[0]+'_obs']
                        elif attr.endswith('obs'):
                            if not np.isnan(data.iloc[i][attr.split('_')[0]+'_M']):
                                data.iloc[i][attr]=data.iloc[i][attr.split('_')[0]+'_M']
                    else:
                        data.iloc[i][attr] = data[attr].mean()
        print(data.info())
        data['psfc_M']=data['psur_M']
        data.drop(['psur_M'],axis=1,inplace=True)
        data.to_csv(os.path.join(opt.src_dir,'nafilled_'+str(j)+'.csv'))
