import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy

MINMAX_DICT = {
    'psur_obs': [850.0, 1100.0],
    't2m_obs': [-20.0, 42.0],
    'q2m_obs': [0.0, 30.0],
    'rh2m_obs': [0.0, 100.0],
    'w10m_obs': [0.0, 30.0],
    'd10m_obs': [0.0, 360.0],
    'u10m_obs': [-20.0, 20.0],
    'v10m_obs': [-20.0, 20.0],
    'RAIN_obs': [0.0, 300.0],
    'psfc_M': [850.0, 1100.0],
    't2m_M': [-20.0, 42.0],
    'q2m_M': [0.0, 30.0],
    'rh2m_M': [0.0, 100.0],
    'w10m_M': [0.0, 30.0],
    'd10m_M': [0.0, 360.0],
    'u10m_M': [-20.0, 20.0],
    'v10m_M': [-20.0, 20.0],
    'SWD_M': [0.0, 1200.0],
    'GLW_M': [0.0, 550.0],
    'HFX_M': [-200.0, 500.0],
    'LH_M': [-50.0, 300.0],
    'RAIN_M': [0.0, 300.0],
    'PBLH_M': [0.0, 5200.0],
    'TC975_M': [-30.0, 40.0],
    'TC925_M': [-35.0, 38.0],
    'TC850_M': [-38.0, 35.0],
    'TC700_M': [-45.0, 30.0],
    'TC500_M': [-70.0, 28.0],
    'wspd975_M': [0.0, 50.0],
    'wspd925_M': [0.0, 50.0],
    'wspd850_M': [0.0, 50.0],
    'wspd700_M': [0.0, 50.0],
    'wspd500_M': [0.0, 60.0],
    'Q975_M': [0.0, 10.0],
    'Q925_M': [0.0, 10.0],
    'Q850_M': [0.0, 10.0],
    'Q700_M': [0.0, 10.0],
    'Q500_M': [0.0, 5.0],
    'psfc_M_pre': [850.0, 1100.0],
    't2m_M_pre': [-20.0, 42.0],
    'q2m_M_pre': [0.0, 30.0],
    'rh2m_M_pre': [0.0, 100.0],
    'w10m_M_pre': [0.0, 30.0],
    'd10m_M_pre': [0.0, 360.0],
    'u10m_M_pre': [-20.0, 20.0],
    'v10m_M_pre': [-20.0, 20.0],
    'SWD_M_pre': [0.0, 1200.0],
    'GLW_M_pre': [0.0, 550.0],
    'HFX_M_pre': [-200.0, 500.0],
    'LH_M_pre': [-50.0, 300.0],
    'RAIN_M_pre': [0.0, 300.0],
    'PBLH_M_pre': [0.0, 5200.0],
    'TC975_M_pre': [-30.0, 40.0],
    'TC925_M_pre': [-35.0, 38.0],
    'TC850_M_pre': [-38.0, 35.0],
    'TC700_M_pre': [-45.0, 30.0],
    'TC500_M_pre': [-70.0, 28.0],
    'wspd975_M_pre': [0.0, 50.0],
    'wspd925_M_pre': [0.0, 50.0],
    'wspd850_M_pre': [0.0, 50.0],
    'wspd700_M_pre': [0.0, 50.0],
    'wspd500_M_pre': [0.0, 60.0],
    'Q975_M_pre': [0.0, 10.0],
    'Q925_M_pre': [0.0, 10.0],
    'Q850_M_pre': [0.0, 10.0],
    'Q700_M_pre': [0.0, 10.0],
    'Q500_M_pre': [0.0, 5.0],
    'hour_sin': [-1.0, 1.0],
    'hour_cos': [-1.0, 1.0],
    'month_sin': [-1.0, 1.0],
    'month_cos': [-1.0, 1.0],
    't2m_prophet':[-20.0,42.0],
    'rh2m_prophet':[0.0,100.0],
    'w10m_prophet':[0.0,30.0],
    'location_90001':[0.,1.0],
    'location_90002':[0.,1.0],
    'location_90003':[0.,1.0],
    'location_90004':[0.,1.0],
    'location_90005':[0.,1.0],
    'location_90006':[0.,1.0],
    'location_90007':[0.,1.0],
    'location_90008':[0.,1.0],
    'location_90009':[0.,1.0],
    'location_90010':[0.,1.0]
}


class Scaler():
    def __init__(self,minmaxdict=None,feature_range=(-1,1)):
        self.minmaxdict=minmaxdict
        self.feature_range=feature_range
        self.minmax_scaler=MinMaxScaler(feature_range=feature_range)
        self.scale_dict={}
        self.min_dict={}
        self.data_range_dict={}


    def transform(self,X):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X must be a DataFrame Object')
        if self.minmaxdict is None:
            return self.minmax_scaler.fit_transform(X)
        else:
            self.cols=X.columns
            res=[]
            for col in self.cols:
                if col not in self.minmaxdict.keys():
                    raise ValueError('donot have minmax value of attr:',col)
                values=X[col].values.copy()
                min_value=self.minmaxdict[col][0]
                max_value=self.minmaxdict[col][1]
                data_range = max_value - min_value
                self.scale_dict[col] = ((self.feature_range[1] - self.feature_range[0]) /(data_range))

                self.min_dict[col] = self.feature_range[0] - min_value * self.scale_dict[col]
                self.data_range_dict[col] = data_range
                values *= self.scale_dict[col]
                values += self.min_dict[col]
                res.append(values)
            res=np.array(res)
            return np.swapaxes(res,0,1)

    def inverse_transform(self,X,cols=None):
        if not isinstance(X,np.ndarray):
            raise ValueError('X must be an ndarray')
        if self.minmaxdict is None:
            return self.minmax_scaler.inverse_transform(X)
        else:
            if cols is not None:
                res=[]
                for col in cols:
                    if col in self.cols:
                        index=list(self.cols).index(col)
                        values=X[:,index]-self.min_dict[col]
                        values/=self.scale_dict[col]
                        res.append(values)
                    else:
                        raise ValueError('col '+col+'doesnot exist')
                return np.swapaxes(np.array(res),0,1)
            else:
                res=copy.deepcopy(X)
                for i in range(len(self.cols)):
                    res[:,i]-=self.min_dict[self.cols[i]]
                    res[:,i]/=self.scale_dict[self.cols[i]]
                return res
