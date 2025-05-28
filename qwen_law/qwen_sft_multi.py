import os
import torch
import wandb
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import find_files_jsonl, formatting_prompts_func_mult,print_trainable_parameters
from typing import Dict, Union, Any, Tuple
import torch
import torch.nn.functional as F

# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TRAIN_SUBSET = 5
EVAL_SUBSET = 1


"""
SFT多轮对话的逻辑和单论对话是不一样的，牵扯到怎么做packing与loss函数计算的不同方式
标准的 SFTTrainer在计算损失时，通常是直接使用模型输出的损失，
拿到 logits 和 labels 后，计算交叉熵损失，并默认对所有有效的（非-100的）标签对应的 token 损失进行平均。
比如三轮对话 我们是要每一轮都计算一次损失 而不是拼在一起后 计算总的损失 
就像：
(l1+l1+l3)/(n1+n2+n3) != (l1/n1+l2/n2+l3/n3)
其中l1表⽰第1个样本的 loss，n1表⽰第1个样本输出的token数量

"""
TMP_PATH = "/hy-tmp/cache" # 确保此路径可写并且有足够空间
DATA_PATH = "/root/llm_post_train/qwen/data/pt"
data_path = "/root/llm_post_train/qwen/data/pt"
output_path = "/root/llm_post_train/qwen/results/pt"
CONFIG_PATH = "/root/llm_post_train/qwen/models/Qwen2.5-0.5B"
model_path = "/root/llm_post_train/qwen/models/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print_trainable_parameters(model)

# 加载数据集并进行预处理
directories = ["duolun"]
data_files = find_files_jsonl(directories,"/root/llm_post_train/qwen/data/")
dataset = load_dataset("json", data_files=data_files, split="train", cache_dir=TMP_PATH) 
dataset = dataset.shuffle(seed=42)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2).values()

if TRAIN_SUBSET > 0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))
if EVAL_SUBSET > 0:
    valid_dataset = valid_dataset.select(range(EVAL_SUBSET))

# 数据整理器
response_template = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
print(response_template_ids)
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

# 训练LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

def get_turn_metadata_for_batch(labels_batch, ignore_index=-100):
    """
    为一批标签计算每个token所属轮次的长度以及批次内的总轮次数。
    Args:
        labels_batch (torch.Tensor): 形状为 [batch_size, seq_len] 的标签张量。
        ignore_index (int): 应该忽略的标签值 (例如 -100)。
    Returns:
        tuple:
            - loss_token_num_batch (torch.Tensor): 形状与 labels_batch 相同，
              每个有效token位置的值是其所属轮次的长度。
            - total_turns_in_batch (torch.Tensor): 该批次中所有样本的总轮次数（标量）。
    """
    batch_size, seq_len = labels_batch.shape
    # 初始化为0，对于无效token或padding，其值为0，不会参与分母计算（因为loss_mask也为0）
    loss_token_num_batch = torch.zeros_like(labels_batch, dtype=torch.float)
    num_turns_per_sample = torch.zeros(batch_size, device=labels_batch.device, dtype=torch.long)

    for i in range(batch_size):  # 遍历批次中的每个样本
        labels_sample = labels_batch[i]
        is_response_token = (labels_sample != ignore_index) # 标记哪些是有效的响应token
        current_turn_len = 0
        turn_start_idx = -1
        sample_turn_count = 0
        for j in range(seq_len): # 遍历序列中的每个token
            if is_response_token[j]: # 如果是有效的响应token
                if current_turn_len == 0: # 新的轮次开始
                    turn_start_idx = j
                    sample_turn_count += 1
                current_turn_len += 1
            else: # 如果不是响应token (是-100或者是序列结束的padding)
                if current_turn_len > 0: # 说明前一个token是一个轮次的结束
                    loss_token_num_batch[i, turn_start_idx : j] = current_turn_len
                    current_turn_len = 0 # 重置当前轮次长度
                    turn_start_idx = -1
        # 如果序列以一个有效的响应轮次结束
        if current_turn_len > 0:
            loss_token_num_batch[i, turn_start_idx : seq_len] = current_turn_len
        num_turns_per_sample[i] = sample_turn_count
    total_turns_in_batch = torch.sum(num_turns_per_sample)
    return loss_token_num_batch, total_turns_in_batch.float()


class CustomLossSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 如果需要，可以在这里添加其他自定义初始化
    # 将辅助函数作为类的方法（或者保持外部独立，然后在这里调用）
    def _get_turn_metadata(self, labels_batch, ignore_index=-100):
        return get_turn_metadata_for_batch(labels_batch, ignore_index)
    def compute_loss(self, model, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        重写损失计算方法。
        """
        # `inputs` 中应该包含由 DataCollatorForCompletionOnlyLM 处理好的 `labels`
        # 我们需要将其取出，不直接传给模型，因为我们要自己计算
        batch_labels = inputs.pop("labels", None)
        if batch_labels is None:
            raise ValueError("在输入中没有找到 'labels'。DataCollatorForCompletionOnlyLM 应该提供它们。")
        # 1. 模型前向传播获取 logits
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if logits is None: 
            if isinstance(outputs, torch.Tensor) and outputs.ndim == logits.ndim: # outputs is logits
                 logits = outputs
            else:
                 raise ValueError("模型输出中没有找到 'logits'。")
        # 2. 计算逐 token 的原始交叉熵损失 (对应您 loss_func 中的 output_tensor)
        # Logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # Labels: (batch_size, seq_len) -> (batch_size * seq_len)
        reshaped_logits = logits.contiguous().view(-1, logits.size(-1))
        reshaped_labels = batch_labels.contiguous().view(-1)    
        # `ignore_index=-100` 会使得对应位置的损失为0，并且不参与后续的梯度计算和平均（如果reduction='mean'）
        # 我们这里用 `reduction='none'`，得到每个token的损失
        per_token_losses_flat = F.cross_entropy(reshaped_logits, reshaped_labels, reduction='none', ignore_index=-100)    
        # 将逐 token 损失 reshape回 [batch_size, seq_len]
        per_token_losses_batch = per_token_losses_flat.view(batch_labels.shape)
        # 3. 创建 loss_mask (哪些 token 实际参与损失计算)
        loss_mask_batch = (batch_labels != -100).float()
        # 4. 获取 loss_token_num_per_token_batch 和 total_turns_in_batch
        loss_token_num_per_token_batch, total_turns_in_batch = self._get_turn_metadata(batch_labels, ignore_index=-100)
        # 5. 应用自定义损失计算逻辑
        # losses * loss_mask (得到有效token的损失)
        masked_token_losses = per_token_losses_batch * loss_mask_batch
        # (losses * loss_mask) / loss_token_num (归一化每个token的损失贡献)
        normalized_loss_contributions = torch.zeros_like(masked_token_losses)
        
        # 创建一个有效进行除法的掩码
        # 即 loss_mask_batch 为 True 且 loss_token_num_per_token_batch 大于 0 的位置
        valid_division_mask = loss_mask_batch.bool() & (loss_token_num_per_token_batch > 0)
        
        normalized_loss_contributions[valid_division_mask] = \
            masked_token_losses[valid_division_mask] / loss_token_num_per_token_batch[valid_division_mask]
        # sum = 1/3 (a2 + a3 + a4) + 1/2 (b1 + b2) + ...
        # 这是对所有 token 的 (loss_token / N_turn_of_token) 进行求和
        # 这等价于对所有轮次的 (AvgLoss_per_turn) 进行求和
        sum_of_normalized_turn_avg_losses = torch.sum(normalized_loss_contributions)
        # 6. 计算最终用于反向传播的损失
        # Trainer 通常期望一个标量损失。可以用批次内的总轮次数来平均这个和。
        if total_turns_in_batch > 0:
            final_loss_for_backprop = sum_of_normalized_turn_avg_losses / total_turns_in_batch
        else:
            # 如果批次中没有有效的轮次（例如，所有标签都是-100，或者批次大小为0）
            # 返回一个需要梯度的0值张量，以避免在训练时出错
            # 使用 logits 的和乘以0，可以确保它连接到计算图，具有正确的设备和类型，并且如果模型参数被使用过，则需要梯度
            final_loss_for_backprop = (logits.sum() * 0.0) 
            # 确保在训练模式下，它确实需要梯度 (通常 logits.sum() * 0.0 已经处理好了)
            if model.training and not final_loss_for_backprop.requires_grad:
                 final_loss_for_backprop = final_loss_for_backprop.clone().requires_grad_(True)
        if return_outputs:
            # SFTTrainer (及基类 Trainer) 期望 (loss, model_outputs_dict)
            log_outputs = outputs.to_dict() if hasattr(outputs, "to_dict") else {"logits": logits} # 确保是字典
            log_outputs["custom_loss_numerator"] = sum_of_normalized_turn_avg_losses.detach().clone()
            log_outputs["custom_loss_denominator"] = total_turns_in_batch.detach().clone()
            return (final_loss_for_backprop, log_outputs)
        return final_loss_for_backprop
# 训练参数配置
training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    eval_steps = 2000,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    save_steps=1000,  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-sft-multi",name="qwen-0.5B-sft-multi"
    )

# 初始化Trainer
trainer = CustomLossSFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # peft_config=lora_config, #是否启用lora
    args=training_args,
    formatting_func= formatting_prompts_func_mult,
    data_collator=collator,
    max_seq_length=512,
    packing=False,
    dataset_num_proc=16,
    dataset_batch_size=5000,
)

# 开始训练
print("Training...")
trainer.train()
trainer.save_model()  
tokenizer.save_pretrained(output_path) 
