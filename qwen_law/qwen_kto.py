import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments 
from trl import KTOConfig, KTOTrainer 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TMP_PATH = "/hy-tmp/kto" 
REPO_PATH = "/root/llm_post_train/qwen" 


output_path_kto = "results/kto"  
data_path_dpo = "data/dpo"       # KTO 可以和 DPO 使用相同的数据集
model_path_sft = "results/sft-1/checkpoint-10000" # SFT model path


model_path = os.path.join(REPO_PATH, model_path_sft)
output_path = os.path.join(REPO_PATH, output_path_kto)
data_path = os.path.join(REPO_PATH, data_path_dpo) 

# Load the SFT model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set pad_token_id if not set, common for Llama-like models
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load the DPO-formatted dataset
data_files = [
    os.path.join(data_path, "test-00000-of-00001.parquet"),
    os.path.join(data_path, "train-00000-of-00001.parquet")
]
dataset = load_dataset("parquet", data_files=data_files, split="train")
dataset = dataset.shuffle(seed=42)

def preprocess_dataset_for_kto(examples):
    """
    预处理 DPO 格式的数据集以适配 KTO。
    每个 DPO 样本(提示、选中、拒绝)会被转换为两个 KTO 样本:
    1. (提示、选中的补全、标签=True) (期望的)
    2. (提示、拒绝的补全、标签=False) (不期望的)
    """
    prompts = []
    completions = []
    labels = [] 

    for i in range(len(examples["prompt"])):

        prompt_text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n"

        assert examples["chosen"][i][1]["role"] == "assistant", \
            f"期望 chosen 的第二条消息角色为 assistant，实际为 {examples['chosen'][i][1]['role']}"
        chosen_completion_text = f"{examples['chosen'][i][1]['content']}<|im_end|>"
        prompts.append(prompt_text)
        completions.append(chosen_completion_text)
        labels.append(True)

        assert examples["rejected"][i][1]["role"] == "assistant", \
            f"期望 rejected 的第二条消息角色为 assistant，实际为 {examples['rejected'][i][1]['role']}"
        rejected_completion_text = f"{examples['rejected'][i][1]['content']}<|im_end|>"
        prompts.append(prompt_text)
        completions.append(rejected_completion_text)
        labels.append(False)

    return {"prompt": prompts, "completion": completions, "label": labels}

original_columns = dataset.column_names
train_dataset_kto = dataset.map(
    preprocess_dataset_for_kto,
    batched=True,
    batch_size=2500, 
    remove_columns=original_columns,
    num_proc=16, 
)


training_args_kto = KTOConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=5e-7,       
    warmup_ratio=0.1,        
    lr_scheduler_type="cosine", 
    num_train_epochs=5,       
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=16, 
    save_steps=500,           
    save_total_limit=10,     
    bf16=True,                
    logging_steps=10,         
    report_to="wandb" if WANDB_LOG else "none",

    beta=0.1,                 # KTO 中的 beta 超参数
    loss_type="kto",          # KTO 的损失类型
    desirable_weight=1.0,     # 正样本权重
    undesirable_weight=1.5,   # 负样本权重

    max_length=1024,          # Max sequence length (prompt + completion)
    max_prompt_length=512,    # Max prompt length

    dataset_num_proc=16, 
    )


if WANDB_LOG:
    try:
        wandb.login() 
        wandb.init(
            project="qwen-0.5B-kto", 
            name="qwen-0.5B-kto-run"  
        )
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Proceeding without W&B logging.")
        training_args_kto.report_to = "none"
        
kto_trainer = KTOTrainer(
    model=model,
    # ref_model=None, # 对于KTO来说,ref_model是可选的。如果为None,策略模型将作为自己的参考模型。
                      # 或者你也可以根据需要加载另一个模型作为ref_model。
                      # 如果precompute_ref_log_probs=True,训练过程中不需要ref_model。
    args=training_args_kto,
    train_dataset=train_dataset_kto,
    tokenizer=tokenizer,
)

kto_trainer.train()

kto_trainer.save_model()
tokenizer.save_pretrained(output_path)

if WANDB_LOG and training_args_kto.report_to == "wandb":
    wandb.finish()

print(f"KTO 训练完成，模型已保存至: {output_path}")
