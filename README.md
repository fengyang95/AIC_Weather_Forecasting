# AI Chanllenger Weather Forecasting 

本方案同时使用seq2seq模型和GBM模型，在[AI Challenger 2018 天气预报 赛题](https://challenger.ai/competition/wf2018)中取得了决赛客观分第1的成绩。


## 开发环境

代码在Windows和Ubuntu系统上测试，使用python3.6。可使用以下命令安装其它依赖库。(其中fbprophet我只在Ubuntu系统安装成功，Windows上的安装比较麻烦)
``` bash
pip install -r requirements.txt
```

## 数据预处理
可在[天气预报赛题](https://challenger.ai/competition/wf2018)官网下载相关训练数据，验证数据以及测试数据。
其中 ai_challenger_wf2018_trainingset_20150301-20180531.nc 存于 data/train 目录，
ai_challenger_wf2018_validation_20180601-20180828_20180905.nc 存于 data/val 目录，
ai_challenger_wf2018_testb7_20180829-20181103.nc 存于 data/testb7 目录。
其他test数据按照类似方法存放。  
1、运行preprocessing目录中的nc2csv_all.bat(或者nc2csv_all.sh)将nc文件处理为csv文件。  
2、运行该目录下 fillna.bat 完成缺失值填充。  
3、运行 generate_prophet_features.py，生成使用fbprophet预测的t2m,rh2m,w10m数据，然后运行
merge_with_prophet.py ，将生成的prophet特征添加到数据文件。  

## 模型训练

本方案采用了两种模型，一种基于RNN的seq2seq方案，一种是基于GBM的方法（分别使用了lightgbm和catboost库）。
所有模型保存在checkpoints目录。
### seq2seq

分别运行 train_on_ts_feature0_global.py、train_on_ts_feature0_global.py、train_on_ts_feature0_global.py
训练得到seq2seq模型（我将自己训练得到的模型存放在checkpoint目录下）。

### GBM
运行train_gbm_all.bat 训练GBM模型。（采用了两种训练方案，一种是对location进行one-hot编码的global方法，只需要训练三个模型即可，另外一种方案
是对每个地点分别训练模型，这样10个地点会有30个模型）


## 模型测试
1、GBM训练的测试信息保存在相应模型目录的feature_importance_info.json文件中。  
2、运行seq2seq目录下的generate_seq2seq_feature48.py，生成seq2seq的结果；运行gbm目录下的
gen_global_all.bat和gen_local_all.bat生成GBM的结果。（相应的日期时间可自行修改）
所有结果保存在result目录。  
3、运行result目录下的merge_all_results.py将所有模型的结果合在一起，最终文件在该目录的merged子目录。
运行caculate_score.py 计算得分(计算方法和评分脚本不完全一致)，得分信息存放在score子目录。
也可运行vis_result.py 查看可视化结果。

## 生成提交文件并验证
```bash
python result/merged/submit/get_submit.py
```
可运行eval目录下的eval_b1_to_b5.bat验证结果。（按照相应规则存放obs,M和submit文件）

## License

This project is licensed under the MIT License
## 参考

1、[IBM seq2seq](https://github.com/IBM/pytorch-seq2seq)  
2、[kaggle-web-traffic 1st place](https://github.com/Arturus/kaggle-web-traffic)

