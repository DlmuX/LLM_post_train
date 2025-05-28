import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)
from transformers.trainer_utils import set_seed


LETTER_LIST = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"
]


def load_model_and_tokenizer(checkpoint_path: str):
    """Load model & tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    return model, tokenizer


def format_example(row: dict, include_answer: bool = True) -> str:
    """格式化为与 MMLU 类似的 prompt。"""
    prompt = "Question: " + row["question"].strip()
    for idx, choice in enumerate(row["choices"]):
        prompt += f"\n{LETTER_LIST[idx]}. {choice.strip()}"
    if include_answer:
        prompt += "\nAnswer: " + row["answer"] + "\n\n"
    else:
        prompt += "\nAnswer:"
    return prompt


def get_choice_tokens(tokenizer, letters: List[str]):
    token_ids = []
    for l in letters:
        # 前面加空格保持与训练一致
        token_ids.extend(tokenizer(" " + l)["input_ids"])
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    return token_ids


def get_logits(tokenizer, model, inputs: List[str], max_seq_len: int):
    inputs_encoded = tokenizer(inputs, padding='longest')["input_ids"]
    input_ids = torch.tensor(inputs_encoded, device=model.device)
    # 截断过长序列
    if input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1] - max_seq_len + 1:]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    outputs = model(input_ids, attention_mask=attention_mask)["logits"]
    logits = outputs[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.no_grad()
def evaluate_truthfulqa(model, tokenizer, ds_df: pd.DataFrame, batch_size: int, max_seq_len: int, few_shot_k: int = 0):
    """对 TruthfulQA 多项选择进行评估。"""
    # few-shot prompt（可选）
    few_shot_prompt = ""
    if few_shot_k > 0:
        sample_df = ds_df.sample(n=few_shot_k, random_state=42)
        for _, r in sample_df.iterrows():
            few_shot_prompt += format_example(r, include_answer=True)

    correct = 0
    total = 0
    pbar_iter = range(0, len(ds_df), batch_size)

    for start in tqdm(pbar_iter, desc="Evaluating"):
        batch_rows = ds_df.iloc[start:start + batch_size].to_dict(orient='records')
        prompts = []
        answers = []
        letters_batch = []
        for r in batch_rows:
            q_prompt = format_example(r, include_answer=False)
            prompts.append(few_shot_prompt + q_prompt)
            answers.append(r["answer"])
            letters_batch.append([LETTER_LIST[i] for i in range(len(r["choices"]))])

        # 获取 logits
        logits = get_logits(tokenizer, model, prompts, max_seq_len)

        # 对每个样本动态 gather 概率
        for i, letters in enumerate(letters_batch):
            # 取每个选项首 token id 作为代表概率
            token_ids = [tokenizer(" " + l)["input_ids"][-1] for l in letters]
            token_ids_tensor = torch.tensor(token_ids, device=model.device)
            probs = logits[i][token_ids_tensor].softmax(-1).detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_letter = letters[pred_idx]
            if pred_letter == answers[i]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def build_dataframe(split_dataset) -> pd.DataFrame:
    data_dicts = []
    for ex in split_dataset:
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        # 有且只有一个正确答案 (mc1)
        correct_idx = labels.index(1) if 1 in labels else None
        if correct_idx is None:
            continue
        data_dicts.append({
            "question": ex["question"],
            "choices": choices,
            "answer": LETTER_LIST[correct_idx]
        })
    return pd.DataFrame(data_dicts)


def main(args):
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)

    dataset = load_dataset("truthful_qa", "multiple_choice")
    # 官方只有 validation split
    split_name = "validation"
    df = build_dataframe(dataset[split_name])

    acc = evaluate_truthfulqa(
        model,
        tokenizer,
        df,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        few_shot_k=args.few_shot
    )
    print("TruthfulQA mc1 Accuracy: {:.2f}%".format(acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA multiple-choice evaluation (mc1)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few-shot", type=int, default=0, help="k-shot examples to include as context")
    args = parser.parse_args()
    set_seed(args.seed)

    main(args) 