import pandas as pd

if __name__=='__main__':
    for i in range(90001,90011):

        prophet_file_path='../data/prophet/prophet_feature_'+str(i)+'.csv'
        prophet_data=pd.read_csv(prophet_file_path,index_col=0)
        prophet_data.index=pd.to_datetime(prophet_data.index)

        train_file_path = '../data/train/merge/nafilled_' + str(i) + '.csv'
        train_data = pd.read_csv(train_file_path, index_col=0)
        train_data.index = pd.to_datetime(train_data.index)
        train_merged_data = pd.merge(train_data, prophet_data, left_index=True, right_index=True)
        train_merged_data.to_csv('../data/train/merge/merged_' + str(i) + '.csv')

        validation_file_path = '../data/val/merge/nafilled_' + str(i) + '.csv'
        validation_data = pd.read_csv(validation_file_path, index_col=0)
        validation_data.index = pd.to_datetime(validation_data.index)
        validation_merged_data = pd.merge(validation_data, prophet_data, left_index=True, right_index=True)
        validation_merged_data.to_csv('../data/val/merge/merged_' + str(i) + '.csv')

        testb7_file_path = '../data/testb7/merge/nafilled_' + str(i) + '.csv'
        testb7_data = pd.read_csv(testb7_file_path, index_col=0)
        testb7_data.index = pd.to_datetime(testb7_data.index)
        testb7_merged_data = pd.merge(testb7_data, prophet_data, left_index=True, right_index=True)
        testb7_merged_data.to_csv('../data/testb7/merge/merged_' + str(i) + '.csv')
