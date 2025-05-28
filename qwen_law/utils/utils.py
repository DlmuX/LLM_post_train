import os
from itertools import chain

def format_to_r1(example):
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def collator_ppo(data):
    # for PPO use
    return {key: [d[key] for d in data] for key in data[0]}

def preprocess_ppo_dataset(examples,tokenizer):
    # for PPO use
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

def preprocess_rm_dataset(examples,tokenizer):
    # for RM use
    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

def format_to_chatml(data):
    formatted_data = []
    for sample in data:
        problem = sample["problem"]
        generation = sample["generation"]
        
        formatted_data.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generation}
            ]
        )
    return {"messages": formatted_data}

def formatting_prompts_func_distill(example):
    # for distill use
    output_texts = []
    for i in range(len(example["problem"])):
        human_text = example["problem"][i]
        gpt_text = example["generation"][i]
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts



def formatting_prompts_func_jsonl_refined(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        instruction = example["instruction"][i].strip() 
        input_text = example["input"][i].strip()   
        output_content = example["output"][i].strip() 

        # 将instruction和input合并为用户的消息
        if input_text: # 检查清理后的 input_text 是否仍有内容
            human_text = f"{instruction}\n{input_text}"
        else:
            human_text = instruction
        
        gpt_text = output_content

        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    
    return output_texts


def formatting_prompts_func_mult(example):
    """
    将单个多轮对话样本（包含 "conversations" 列表）转换为多个SFT训练文本序列。
    每个序列包含到某个助手回答为止的完整对话历史。

    Args:
        example (dict): 包含 "conversations"键的字典，
                        其值为一个列表，列表中的每个元素是形如
                        {"role": "user/assistant", "content": "..."} 的字典。

    Returns:
        list: 包含一个或多个SFT训练文本字符串的列表。
    """
    output_texts = []
    for i in range(len(example["conversations"])):
        text = ""
        for item in example["conversations"][i]:
            if item["from"] == "human":
                human_text = item["value"]
                text += f"<|im_start|>user\n{human_text}<|im_end|>"
            elif item["from"] == "gpt":
                gpt_text = item["value"]
                text += f"<|im_start|>assistant\n{gpt_text}<|im_end|>"
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        output_texts.append(text)
    return output_texts

def find_files(dirs,path="data/pt"):
    """
    遍历目录，查找所有文件
    """
    files = []
    for dir in dirs:
        base_path = os.path.join(path, dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

def find_files_jsonl(dirs, path="data/pt"):
    """
    遍历目录，查找所有 .jsonl 文件
    """
    files = []
    for dir_item in dirs: # 将 'dir' 修改为 'dir_item' 以避免与内置函数 dir 冲突
        base_path = os.path.join(path, dir_item)
        
        # 增加一个检查，确保 base_path 是一个存在的目录
        if not os.path.isdir(base_path):
            print(f"警告: 路径 '{base_path}' 不是一个有效的目录，已跳过。")
            continue
            
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".jsonl"): # 这里是主要的修改点
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

"""
1. Random Concatenate (随机拼接)
"""
def tokenize_dataset(examples,tokenizer,block_size = 512):
    """
    预处理预训练数据集，将文本分词并分块
    """
    eos_token = "<|im_end|>"
    # examples["text"] = ["第一段短文。", "第二段是中等长度的文本。", "最后一段很长很长。"]
    text_examples = [text + eos_token for text in examples["text"]]
    # text_examples 会变成:
    # [
    #   "第一段短文。<|im_end|>",
    #   "第二段是中等长度的文本。<|im_end|>",
    #   "最后一段很长很长。<|im_end|>"
    # ]
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    # tokenizer 会将每个字符串转换为 token ID 列表
    # add_special_tokens=False 确保 tokenizer 不会额外添加它自己的特殊 token (如 [CLS], [SEP])
    # tokenized_examples =
    # {
    #   'input_ids': [
    #     [10, 20, 30, 99],                             # "第一段短文。<|im_end|>" (长度4)
    #     [40, 50, 60, 70, 80, 99],                     # "第二段是中等长度的文本。<|im_end|>" (长度6)
    #     [11, 22, 33, 44, 55, 66, 77, 88, 99]          # "最后一段很长很长。<|im_end|>" (长度9)
    #   ]
    # }
    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    # chain函数就是起到压平作用 
    # concatenated_examples = {
    # 'input_ids': [10, 20, 30, 99, 40, 50, 60, 70, 80, 99, 11, 22, 33, 44, 55, 66, 77, 88, 99],
    # }
    
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    # 4. 计算总长度
    # 取第一个键 (通常是 'input_ids') 来确定总 token 数
    # total_length = 19

    total_length = (total_length // block_size) * block_size
    # 5. 对齐块大小，丢弃末尾不足一个块的部分
    # block_size = 10
    # (19 // 10) * 10  =>  1 * 10  =>  10

     # 6. 将长序列分割成固定长度的块
    # 从拼接后的序列中，每 block_size 个 token 取一个块，只取到对齐后的 total_length
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    return result, total_length

"""
Random Concatenate + NoiseMask (随机拼接 + 噪音掩码)
"""

def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    

def tokenize_dataset2(examples,tokenizer,block_size=512):
    """
    预处理预训练数据集，将文本分词并分块
    """
    eos_token = "<|im_end|>"
    text_examples = [text + eos_token for text in examples["text"]]  # 添加结束符
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)

    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // block_size) * block_size  # 对齐块大小

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result,total_length