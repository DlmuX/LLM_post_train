#!/usr/bin/env bash
# 下载 TruthfulQA 多项选择数据集，并转换为 csv 方便评估脚本使用

set -e

# 目标目录
DATA_DIR="data/truthful_qa"
mkdir -p ${DATA_DIR}

python - << 'PY'
from datasets import load_dataset
import pandas as pd
import os
import string

LETTER_LIST = list(string.ascii_uppercase)

ds = load_dataset("truthful_qa", "multiple_choice")

os.makedirs("data/truthful_qa", exist_ok=True)

def save_split(split_name):
    split_ds = ds[split_name]
    records = []
    for ex in split_ds:
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        try:
            correct_idx = labels.index(1)
        except ValueError:
            # 如果没有正确答案，跳过
            continue
        record = {
            "question": ex["question"],
            "answer": LETTER_LIST[correct_idx]
        }
        for idx, choice in enumerate(choices):
            record[LETTER_LIST[idx]] = choice
        records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(f"data/truthful_qa/{split_name}.csv", index=False)

save_split("validation")  # TruthfulQA 只有 validation split
print("TruthfulQA 数据已保存至 data/truthful_qa")
PY 