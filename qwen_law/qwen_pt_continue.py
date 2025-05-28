import os
import torch 
from datasets import load_dataset,Dataset,concatenate_datasets
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,    
)
from utils.utils import find_files,tokenize_dataset

TRUNK_SIZE = 512
TMP_PATH = "/hy-tmp/cache" # 确保此路径可写并且有足够空间
DATA_PATH = "data/pt"
OUTPUT_PATH = "results/pt"
CONFIG_PATH = "models/Qwen2.5-0.5B"
WANDB_LOG = True # 如果不想记录到 W&B，请设置为 False

# 最好在 PyTorch 初始化 CUDA 之前尽早设置此环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# --- 路径和模型设置 ---
output_path = OUTPUT_PATH
model_path = CONFIG_PATH

print(f"从以下路径加载模型配置: {model_path}")
config = AutoConfig.from_pretrained(model_path)

print(f"从以下路径加载分词器: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 如果 pad_token 未设置，则进行设置，这对于类 GPT 模型很常见
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = config.eos_token_id 

print("从配置初始化模型...")
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

model.resize_token_embeddings(len(tokenizer))

# 最终合并并分词后的训练数据集的路径
tokenized_datapath_combined = os.path.join(DATA_PATH, "tokenized_dataset_combined_for_training")

# 定义需要单独监控的类别 为了后面channel loss监控 这里可以改成自己的数据集文件类别
categories = [
    "mathematics_statistics",
    "law_judiciary",
]

# 用于存放分词后评估数据集的字典
eval_datasets_for_trainer = {}
# 用于存放在合并为训练集之前各个分词后数据集的列表
all_tokenized_datasets_for_combination = []

# 定义用于分词的映射回调函数
def map_callback(examples):
    # 调用了utils包下的tokenize_dataset函数 返回一个包含 'input_ids'、'attention_mask' 和 'labels' 的字典
    result, _ = tokenize_dataset(examples, tokenizer, TRUNK_SIZE)
    return result

#处理每个类别
for category in categories:
    tokenized_category_path = os.path.join(DATA_PATH,f"tokenized_{category}_eval")
    print(f"处理列表：{category}")
    if not os.path.isdir(tokenized_category_path):
        print(f"  类别 '{category}' 的分词数据在路径 {tokenized_category_path} 未找到。正在处理...")
        category_data_files = find_files([category])
        if not category_data_files:
            print(f"  警告: 使用 find_files(['{category}']) 未找到类别 '{category}' 的数据文件。正在跳过。")
            continue
        print(f"  从文件加载类别 '{category}' 的原始数据集: {category_data_files}")
        try:
            dataset_category_raw = load_dataset(
                "parquent",
                data_files = category_data_files,
                split = "train",
                columns = ["text"],
                cache_dir = TMP_PATH
            )
        except Exception as e:
            print(f"  加载类别 {category} 的数据集时出错: {e}。请检查 Parquet 文件和 'text' 列。文件: {category_data_files}")
            continue
    
    # 根据CPU核心数和内存调整 num_proc 这个过程非常吃内存 项目训练中5个G的数据分完词在80G左右。
        print(f"  正在分词 '{category}'...")
        num_processors = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    
        #这个函数很重要 无论是预训练还是微调，语言模型都需要数值化的输入，而不是原始文本
        tokenizer_category_dataset = dataset_category_raw.map(
            map_callback,
            batched = True,
            batch_size = 5000,
            remove_columns=dataset_category_raw.column_names,
            num_proc=num_processors,
            )
        print(f"  正在将分词后的 '{category}' 保存到磁盘: {tokenized_category_path}")
        tokenized_category_dataset.save_to_disk(tokenized_category_path)
    else:
        print(f"  从磁盘加载分词后的 '{category}': {tokenized_category_path}")
        tokenized_category_dataset = Dataset.load_from_disk(tokenized_category_path)
        
# 从所有类别创建合并的训练数据集
if not os.path.isdir(tokenized_datapath_combined):
    if not all_tokenized_datasets_for_combination:
        raise ValueError("没有加载或处理任何分词后的数据集。无法创建合并的训练数据集。")
    print(f"正在合并所有分词后的数据集用于训练，并保存到 {tokenized_datapath_combined}...")
    train_dataset = concatenate_datasets(all_tokenized_datasets_for_combination)
    print("正在打乱合并后的训练数据集...")
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset.save_to_disk(tokenized_datapath_combined)
else:
    print(f"从 {tokenized_datapath_combined} 加载合并的分词后训练数据集...")
    train_dataset = Dataset.load_from_disk(tokenized_datapath_combined)
    
print(f"合并后的训练数据集中的总样本数: {len(train_dataset)}")
for name, ds in eval_datasets_for_trainer.items():
    print(f"评估数据集 '{name}' 中的总样本数: {len(ds)}")

# --- 整理器和训练参数 ---
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=12, 
    gradient_accumulation_steps=16, # 这个参数要了解一下 和梯度更新有很大关系 不同数据集规模下 这个参数的大小也不同 有效批处理大小 = (per_device_train_batch_size * gradient_accumulation_steps * GPU数量)
    save_strategy="steps",          
    save_steps=10_000,
    save_total_limit=3,
    gradient_checkpointing=True,
    bf16=True, 
    logging_steps=50, # 记录训练损失
    report_to="wandb" if WANDB_LOG else "none",
    evaluation_strategy="steps", #定期进行评估
    eval_steps=500, # 每 N 步评估一次。
    # metric_for_best_model="loss",# 如果为 eval_dataset 提供了字典，则默认为第一个评估数据集的损失
)

# --- W&B 设置 ---
if WANDB_LOG:
    try:
        wandb.login() # 尝试登录，将使用现有登录信息或在需要时提示
        wandb.init(
            project="qwen-0.5B-pt-multi-eval", 
            name="qwen-0.5B-pt-run", 
            config=training_args.to_dict() 
        )
        wandb.config.update({
            "model_path": model_path,
            "trunk_size": TRUNK_SIZE,
            "data_path": DATA_PATH,
            "categories": categories,
        })
    except Exception as e:
        print(f"W&B 初始化失败: {e}。正在禁用 W&B 日志记录。")
        WANDB_LOG = False
        training_args.report_to = "none" # 确保 Trainer 不会尝试上报
        
        
# --- Trainer 初始化 ---
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets_for_trainer if eval_datasets_for_trainer else None, # 传递评估数据集字典
    tokenizer=tokenizer, 
)

# --- 训练 ---
print("训练前清除 CUDA 缓存...")
torch.cuda.empty_cache()

print("开始训练...")
try:
    trainer.train()
except Exception as e:
    print(f"训练过程中发生错误: {e}")
    if WANDB_LOG and wandb.run is not None:
        wandb.log({"training_error": str(e)}) 
    raise 

print("训练完成。正在保存最终模型...")
trainer.save_model() # 将模型保存到 training_args.output_dir

tokenizer.save_pretrained(output_path)
print(f"模型和分词器已保存到 {output_path}")

if WANDB_LOG and wandb.run is not None:
    wandb.finish()

print("脚本完成。")