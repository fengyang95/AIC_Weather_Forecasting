import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
MEANSTD_DICT=\
    {'psfc_M': [984.3723478329193, 31.738102557857424],
     't2m_M': [12.16030384913689, 12.477564966324843],
     'q2m_M': [5.607230253135884, 4.811316603716048],
     'rh2m_M': [48.268706534587295, 20.8947943762082],
     'w10m_M': [2.5059690958760594, 1.6922711438512235],
     'd10m_M': [183.40992340835143, 90.80565823281526],
     'u10m_M': [0.2700926431013989, 1.9783868971730012],
     'v10m_M': [0.1387687383995052, 2.1106814357561037],
     'SWD_M': [183.47554534924546, 271.6910000151648],
     'GLW_M': [294.0872718427513, 79.17290947167352],
     'HFX_M': [34.29115049455108, 80.30589081532823],
     'LH_M': [41.83030098806697, 73.33590703880263],
     'RAIN_M': [1.070807605157258, 13.99437296667716],
     'PBLH_M': [615.2981309283356, 694.5626243560781],
     'TC975_M': [12.556855326647302, 12.005631062419504],
     'TC925_M': [9.688404535759393, 11.853407351435967],
     'TC850_M': [5.334843311715559, 11.31516533027558],
     'TC700_M': [-3.7120842241936436, 9.881915741129959],
     'TC500_M': [-19.19023232517952, 9.049613830143558],
     'wspd975_M': [9.975807200525775, 10.062799427284173],
     'wspd925_M': [6.745719492085971, 4.345051051784165],
     'wspd850_M': [7.96582604877283, 4.509258305479629],
     'wspd700_M': [10.670579237907287, 5.179787383436679],
     'wspd500_M': [17.778019554193385, 9.162205746715026],
     'Q975_M': [3.785280556405308, 323.64778239874096],
     'Q925_M': [5.504728662728674, 323.6435696602653],
     'Q850_M': [5.248671273811768, 323.63977220400386],
     'Q700_M': [3.5197517409140127, 323.626460163696],
     'Q500_M': [1.6457266591471371, 323.61851264589325],
     'psur_obs': [982.5711800294754, 40.64783373285083],
     't2m_obs': [11.508868546576798, 12.085801427316497],
     'q2m_obs': [6.024612924697347, 5.092986199807948],
     'w10m_obs': [2.170412529505236, 1.6323531115560295],
     'd10m_obs': [172.80460088594586, 105.94074397794164],
     'rh2m_obs': [53.596920594837265, 25.77578890057738],
     'u10m_obs': [0.15443169796385373, 1.9060829955003524],
     'v10m_obs': [-0.16772603364134053, 1.9165836616766607],
     'RAIN_obs': [0.06598099046015816, 0.7717128324057292],
     't2m_prophet': [11.50884535211354, 11.543132604429479],
     'rh2m_prophet': [53.59709389676056, 18.453271593619796],
     'w10m_prophet': [2.170416110144887, 0.9614022509656095],
    'hour_sin': [0.0, 1.0],
     'hour_cos': [0.0, 1.0],
    'month_sin': [0.0, 1.0],
    'month_cos': [0.0, 1.0],
     }



class StdScaler():
    def __init__(self,meanstd_dict=None):
        self.standard_scaler=StandardScaler()
        self.meanstd_dict=meanstd_dict


    def transform(self,X):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X must be a DataFrame Object')
        if self.meanstd_dict is None:
            return self.standard_scaler.fit_transform(X)
        else:
            self.cols=X.columns
            res=[]
            for col in self.cols:
                if col not in self.meanstd_dict.keys():
                    raise ValueError('donot have mean std value of attr:',col)
                values=X[col].values.copy()
                mean_value=self.meanstd_dict[col][0]
                std_value=self.meanstd_dict[col][1]
                values -= mean_value
                values /= std_value
                res.append(values)
            res=np.array(res)
            return np.swapaxes(res,0,1)

    def inverse_transform(self,X,cols=None):
        if not isinstance(X,np.ndarray):
            raise ValueError('X must be an ndarray')
        if self.meanstd_dict is None:
            return self.standard_scaler.inverse_transform(X)
        else:
            if cols is not None:
                res=[]
                for col in cols:
                    if col in self.cols:
                        index=list(self.cols).index(col)
                        values=X[:,index]*self.meanstd_dict[col][1]
                        values+=self.meanstd_dict[col][0]
                        res.append(values)
                    else:
                        raise ValueError('col '+col+'doesnot exist')
                return np.swapaxes(np.array(res),0,1)
            else:
                res=copy.deepcopy(X)
                for i in range(len(self.cols)):
                    res[:,i]*=self.meanstd_dict[self.cols[i]][1]
                    res[:,i]+=self.meanstd_dict[self.cols[i]][0]
                return res
