import pandas as pd
import numpy as np
import os
import catboost
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.externals import joblib
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
import json
import argparse

# 加上48小时前obs 信息
#  处理u=u/w v=v/w 信息 ，去掉重复的 d

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', dest='log_level', default='info', type=str,
                        help='Logging level.')
    parser.add_argument('--boost_type', dest='boost_type',
                        default='lgb',
                        type=str, help='boost type can be "lgb" or "catboost" ')
    parser.add_argument('--with_Q_feature', dest='Q', default=True, type=str2bool, help='whether to use Q_ features')

    opt = parser.parse_args()

    feature_columns = ['t2m_obs', 'rh2m_obs', 'w10m_obs', 'psur_obs', 'q2m_obs', 'u10m_obs',
                       'v10m_obs', 'RAIN_obs',
                       't2m_prophet', 'rh2m_prophet', 'w10m_prophet',
                       't2m_M', 'rh2m_M', 'w10m_M', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                       'psfc_M', 'q2m_M', 'u10m_M', 'v10m_M',
                       'SWD_M', 'GLW_M', 'HFX_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M',
                       'TC850_M', 'TC700_M', 'TC500_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M']
    if opt.Q is True:
        feature_columns += ['Q975_M', 'Q925_M', 'Q850_M', 'Q700_M', 'Q500_M', 'LH_M']

    all_features = feature_columns[8:] + list(str(i) for i in range(-24, 0)) + \
                   [(attr + '_48ago') for attr in ['t2m_obs', 'rh2m_obs', 'w10m_obs', 'psur_obs', 'q2m_obs', 'u10m_obs',
                                                   'v10m_obs', 'RAIN_obs']
                    ]

    print(len(all_features))
    feature_importance_dict = {}
    feature_importance_dict['t2m'] = {}
    feature_importance_dict['rh2m'] = {}
    feature_importance_dict['w10m'] = {}
    feature_importance_dict['train_score'] = {}
    feature_importance_dict['val_score'] = {}
    feature_importance_dict['ratio'] = {}

    feature_importance_dict['train_score']['t2m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['train_score']['rh2m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['train_score']['w10m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['val_score']['t2m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['val_score']['rh2m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['val_score']['w10m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['ratio']['t2m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['ratio']['rh2m'] = {'score': [], 'mean': 0.0}
    feature_importance_dict['ratio']['w10m'] = {'score': [], 'mean': 0.0}

    for feature in all_features:
        feature_importance_dict['t2m'][feature] = []
        feature_importance_dict['rh2m'][feature] = []
        feature_importance_dict['w10m'][feature] = []
    target_list = ['t2m', 'rh2m', 'w10m']


    history_num = 24
    model_dir = None

    if opt.boost_type == "lgb":
        model_dir = '../checkpoints/lgb'
    elif opt.boost_type == "catboost":
        model_dir = '../checkpoints/catboost'
    else:
        raise ValueError('invalid boost type')
    if opt.Q is True:
        model_dir += '_q'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for i in range(90001,90011):
        for feature_index in range(3):
            train_data = pd.read_csv('../data/train/merge/merged_' + str(i) + '.csv', index_col=0)
            train_data.index = pd.to_datetime(train_data.index)
            train_data = train_data['2015-03-02':]
            train_data['hour'] = train_data.index.hour
            train_data['month'] = train_data.index.month
            hour_period = 24 / (2 * np.pi)
            train_data['hour_cos'] = np.cos(train_data.index.hour / hour_period)
            train_data['hour_sin'] = np.sin(train_data.index.hour / hour_period)

            month_period = 12 / (2 * np.pi)
            train_data['month_cos'] = np.cos(train_data.index.month / month_period)
            train_data['month_sin'] = np.sin(train_data.index.month / month_period)

            train_data['u10m_obs']=train_data['u10m_obs']/train_data['w10m_obs']
            train_data['v10m_obs']=train_data['v10m_obs']/train_data['w10m_obs']
            train_data['u10m_M']=train_data['u10m_M']/train_data['w10m_M']
            train_data['v10m_M']=train_data['v10m_M']/train_data['w10m_M']

            train_data = pd.DataFrame(train_data, columns=feature_columns)

            for col in train_data.columns:
                train_data[col] = train_data[col].fillna(train_data[col].mean())

            train_float_data = np.array(train_data)

            tr_float_data = train_float_data[48:,:]
            length_tr = tr_float_data.shape[0]
            tr_history_24_hours = np.zeros((length_tr, history_num))
            for k in range(history_num):
                tr_history_24_hours[:, k] = train_float_data[24+k:k - history_num, feature_index]
            tr_obs_48hours_ago=train_float_data[:-48,:8]

            whole_train_float_data = np.c_[tr_float_data, tr_history_24_hours,tr_obs_48hours_ago]
            # whole_train_float_data=tr_float_data

            validation_data = pd.read_csv('../data/val/merge/merged_' + str(i) + '.csv', index_col=0)
            validation_data.index = pd.to_datetime(validation_data.index)
            validation_data = validation_data['2018-06-03':]
            validation_data['hour'] = validation_data.index.hour
            validation_data['month'] = validation_data.index.month
            hour_period = 24 / (2 * np.pi)
            validation_data['hour_cos'] = np.cos(validation_data.index.hour / hour_period)
            validation_data['hour_sin'] = np.sin(validation_data.index.hour / hour_period)

            month_period = 12 / (2 * np.pi)
            validation_data['month_cos'] = np.cos(validation_data.index.month / month_period)
            validation_data['month_sin'] = np.sin(validation_data.index.month / month_period)

            validation_data['u10m_obs'] = validation_data['u10m_obs'] / validation_data['w10m_obs']
            validation_data['v10m_obs'] = validation_data['v10m_obs'] / validation_data['w10m_obs']
            validation_data['u10m_M'] = validation_data['u10m_M'] / validation_data['w10m_M']
            validation_data['v10m_M'] = validation_data['v10m_M'] / validation_data['w10m_M']

            validation_data = pd.DataFrame(validation_data,
                                           columns=feature_columns)

            for col in validation_data.columns:
                validation_data[col] = validation_data[col].fillna(validation_data[col].mean())

            validation_float_data = np.array(validation_data)

            val_float_data = validation_float_data[48:, :]
            length_val = val_float_data.shape[0]
            val_history_24_hours = np.zeros((length_val, history_num))
            for k in range(history_num):
                val_history_24_hours[:, k] = validation_float_data[k+24:k - history_num, feature_index]
            val_obs_48hours_ago = validation_float_data[:-48, :8]
            whole_validation_data = np.c_[val_float_data, val_history_24_hours,val_obs_48hours_ago]

            X_train = whole_train_float_data[:,8:]
            y_train = whole_train_float_data[:, feature_index]
            y_train_M=whole_train_float_data[:,11+feature_index]

            X_test = whole_validation_data[:,8:]
            y_test = whole_validation_data[:, feature_index]
            y_test_M = whole_validation_data[:, 11 + feature_index]

            model=None
            if opt.boost_type=='lgb':
                params = {
                #'reg_alpha': [0, 20, 40],
                #'reg_lambda': [0, 20, 40]
                'subsample_for_bin':[20000,2000,1000],
                'num_leaves':[31,15,50],
                }
                gs_cv = GridSearchCV(param_grid=params, estimator=lgb.LGBMRegressor(), cv=5, verbose=True)
                gs_cv.fit(X_train, y_train)
                print(gs_cv.best_params_)
                print(gs_cv.best_score_)
                joblib.dump(gs_cv.best_estimator_,
                        os.path.join(model_dir, 'model_' + str(i) + '_feature_' + str(feature_index) + '.m'))

                model = gs_cv.best_estimator_
            elif opt.boost_type=='catboost':
                model=catboost.CatBoostRegressor()
                model.fit(X_train,y_train)
                joblib.dump(model,
                            os.path.join(model_dir, 'model_' + str(i) + '_feature_' + str(feature_index) + '.m'))

            print(model.feature_importances_)

            for ii, feature in enumerate(all_features):
                feature_importance_dict[target_list[feature_index]][feature].append(
                    float(model.feature_importances_[ii]))

            y_train_pred=model.predict(X_train)
            y_test_pred = model.predict(X_test)

            rmse_M = np.sqrt(mean_squared_error(y_test, y_test_M))
            rmse_FORE = np.sqrt(mean_squared_error(y_test, y_test_pred))

            train_rmse_M=np.sqrt(mean_squared_error(y_train,y_train_M))
            train_rmse_FORE=np.sqrt(mean_squared_error(y_train,y_train_pred))

            train_score=-(train_rmse_FORE-train_rmse_M)/train_rmse_M

            val_score=-(rmse_FORE-rmse_M)/rmse_M

            feature_importance_dict['train_score'][target_list[feature_index]]['score'].append(train_score)
            feature_importance_dict['val_score'][target_list[feature_index]]['score'].append(val_score)
            feature_importance_dict['ratio'][target_list[feature_index]]['score'].append(val_score/train_score)
            print('XGBR:')
            print('rmse_M:', rmse_M)
            print('rmse_FORE:', rmse_FORE)
    for key in all_features:
        feature_importance_dict['t2m'][key]=np.mean(np.array(feature_importance_dict['t2m'][key]))
        feature_importance_dict['rh2m'][key]=np.mean(np.array(feature_importance_dict['rh2m'][key]))
        feature_importance_dict['w10m'][key]=np.mean(np.array(feature_importance_dict['w10m'][key]))
    for obs in target_list:
        feature_importance_dict[obs] = OrderedDict((k, v) for k, v in sorted(feature_importance_dict[obs].items(), key=lambda x: x[1],reverse=True))

        train_score_list=feature_importance_dict['train_score'][obs]['score']
        val_score_list=feature_importance_dict['val_score'][obs]['score']
        ratio_list=feature_importance_dict['ratio'][obs]['score']
        feature_importance_dict['train_score'][obs]['mean']=np.mean(np.array(train_score_list))
        feature_importance_dict['val_score'][obs]['mean']=np.mean(np.array(val_score_list))
        feature_importance_dict['ratio'][obs]['mean']=np.mean(np.array(ratio_list))
    feature_importance_dict['train_score']['mean']=(feature_importance_dict['train_score']['t2m']['mean']+
                                                    feature_importance_dict['train_score']['rh2m']['mean']+
                                                    feature_importance_dict['train_score']['w10m']['mean'])/3
    feature_importance_dict['val_score']['mean']=(feature_importance_dict['val_score']['t2m']['mean']+
                                                  feature_importance_dict['val_score']['rh2m']['mean']+
                                                  feature_importance_dict['val_score']['w10m']['mean'])/3
    feature_importance_dict['ratio']['mean']=(feature_importance_dict['ratio']['t2m']['mean']+
                                              feature_importance_dict['ratio']['rh2m']['mean']+
                                              feature_importance_dict['ratio']['w10m']['mean'])/3


    json_content = json.dumps(feature_importance_dict)
    f = open(os.path.join(model_dir,'feature_importance_info.json'), 'w')
    f.write(json_content)
    f.close()



















