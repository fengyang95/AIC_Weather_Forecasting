import pandas as pd
from fbprophet import Prophet

if __name__=='__main__':
   for i in range(90001,90011):
       data = pd.read_csv('../data/train/nafilled_'+str(i)+'.csv')
       data['Time'] = pd.to_datetime(data['Time'])
       target_csv = None
       for attr in ['t2m_obs', 'rh2m_obs', 'w10m_obs']:
           tmp_df = data.loc[:, ['Time', attr]]
           tmp_df.columns = ['ds', 'y']
           m = Prophet(weekly_seasonality=False, yearly_seasonality=False)
           m.fit(tmp_df)
           future = m.make_future_dataframe(periods=5000, freq='H')
           forecast = m.predict(future)
           proph = forecast[['yhat','yhat_lower','yhat_upper']]
           proph.columns = [attr.split('_')[0] + '_prophet',attr.split('_')[0]+'_prophet_lower',
                            attr.split('_')[0]+'_prophet_upper']
           if target_csv is None:
               target_csv = forecast[['ds']]
               target_csv.columns = ['Time']
           target_csv = pd.concat([target_csv, proph], axis=1)
           print(attr + ' Done!')
       target_csv.to_csv('prophet_'+str(i)+'.csv',index=False)

