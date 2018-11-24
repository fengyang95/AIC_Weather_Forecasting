
python nc2csv.py --src_file_path=../data/train/ai_challenger_wf2018_trainingset_20150301-20180531.nc --str_lastday=20180601 --dst_dir=../data/train/merge --method=merge

python nc2csv.py --src_file_path=../data/train/ai_challenger_wf2018_trainingset_20150301-20180531.nc  --dst_dir=../data/train/obs_and_M --method=obs_and_M


python nc2csv.py --src_file_path=../data/val/ai_challenger_wf2018_validation_20180601-20180828_20180905.nc --str_lastday=20180829 --dst_dir=../data/val/merge --method=merge


python nc2csv.py --src_file_path=../data/val/ai_challenger_wf2018_validation_20180601-20180828_20180905.nc  --dst_dir=../data/val/obs_and_M --method=obs_and_M


python nc2csv.py --src_file_path=../data/testb7/ai_challenger_wf2018_testb7_20180829-20181103.nc --str_lastday=20181104 --dst_dir=../data/testb7/merge --method=merge


python nc2csv.py --src_file_path=../data/testb7/ai_challenger_wf2018_testb7_20180829-20181103.nc  --dst_dir=../data/testb7/obs_and_M --method=obs_and_M
