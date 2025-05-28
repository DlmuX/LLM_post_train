modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ./models/Qwen2.5-0.5B

## 增量预训练数据集 数据进行map后会很大，我实验的5G数据大概需要100G存储吧，所以想跑一下试试的 我建议下载两到三个就可以 然后下载两到三类数据集 (这里可以考虑一下为什么)
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'mathematics_statistics/chinese/high/rank_00068.parquet' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'law_judiciary/chinese/high/rank_00058.parquet' --local_dir 'data/pt'
modelscope download --dataset 'BAAI/IndustryCorpus2' --include 'law_judiciary/chinese/high/rank_00059.parquet' --local_dir 'data/pt'  


# 下载评估数据集MMLU
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
tar xf data/mmlu/data.tar -C data/mmlu

# 下载指令微调训练数据集
modelscope download --dataset 'BAAI/Infinity-Instruct' --local_dir 'data/sft' # 选择7M和Gen进行微调，因为这两个数据集更新时间最近，且数据量大

# 下载偏好数据集
modelscope download --dataset 'BAAI/Infinity-Preference' --local_dir 'data/dpo'

# 下载蒸馏数据集
modelscope download --dataset HuggingFaceH4/numina-deepseek-r1-qwen-7b --local_dir 'data/distill'

# 下载偏好数据集
modelscope download --dataset swift/stack-exchange-paired --local_dir 'data/reward'

# 下载Reasoning数据集
modelscope download --dataset AI-MO/NuminaMath-TIR --local_dir 'data/reasoning'