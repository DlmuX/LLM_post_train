# Qwen-Law: 法律领域大语言模型

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.45+-green.svg)](https://huggingface.co/transformers/)

> 🏛️ 本项目致力于在法律领域构建和优化大语言模型，涵盖了从数据准备、增量预训练、指令微调到强化学习对齐的完整流程。

## 📋 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
  - [增量预训练 (PT)](#增量预训练-pt)
  - [指令微调 (SFT)](#指令微调-sft)
  - [强化学习对齐](#强化学习对齐)
- [模型评估](#模型评估)
- [推理与聊天机器人](#推理与聊天机器人)
- [日志与监控](#日志与监控)
- [其他脚本](#其他脚本)

## 🔧 环境配置

### Python 环境
建议使用 Python 3.8 或更高版本。

### 依赖安装
- 安装核心依赖（不包括 GRPO 特有依赖）：
```bash
pip install -r requirements.txt
```
- 安装 GRPO 相关依赖：
```bash
pip install -r requirements_grpo.txt
```

## 📊 数据准备

### 数据下载
所有需要的数据集下载脚本已整合到 `download_data.sh`。
```bash
bash download_data.sh
```
> **⚠️ 注意**: 如果需要分批次下载或选择性下载特定数据集，请修改 `download_data.sh` 脚本中的相关注释和命令。

## 🚀 模型训练

模型训练流程主要包括增量预训练、指令微调和强化学习对齐。

### 增量预训练 (PT)

在通用基座模型的基础上，使用法律领域数据进行增量预训练，以增强模型在法律领域的知识和理解能力。

#### 单GPU训练
```bash
CUDA_VISIBLE_DEVICES=0 python mini_qwen_pt.py \
    --model_name_or_path <base_model_path> \
    --train_data_path <pretrain_data_path> \
    --output_dir <pt_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### 多GPU训练 (Accelerate)
```bash
accelerate launch --config_file accelerate_config.yaml qwen_pt.py \
    --model_name_or_path <base_model_path> \
    --train_data_path <pretrain_data_path> \
    --output_dir <pt_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### 📝 训练精度说明
- **混合精度训练**: 全精度加载模型，并在 `training_args` 中设置 `fp16 = True` (即命令行使用 `--fp16` 参数)。
- **单精度训练**: 全精度加载模型，`training_args` 中 `fp16 = False` (即命令行不使用 `--fp16` 或 `--bf16` 参数)。

#### 🎯 考点
增量预训练这部分主要做好数据配比问题，一般为预训练数据的10%到20%，同时，通用数据，专用领域数据，代码和数学等思维知识的一个配比。这里面有一个技巧是channel loss，分别监控在这三个数据集上的loss表现，可以在评测前去做一些判断和及时调整数据的配比。同时也要知道，增量预训练的作用，和预训练，sft微调的区别在哪里。

比如：
- **增量预训练**: 目标是让模型知识更渊博，更能理解和生成某一类文本。Loss的计算反映了模型对整个文本序列的理解和预测能力。
- **SFT**: 是给模型"上课、做练习题"，教它如何根据问题给出正确的答案。Loss的计算只关注答案部分是否正确，不关心它是否能预测问题本身。

### 指令微调 (SFT)

使用高质量的指令数据对预训练后的模型进行微调，使其能够更好地理解和遵循用户指令。

#### 单轮对话微调

针对单轮问答、指令跟随等场景进行微调。

**单GPU训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python qwen_sft_single_turn.py \
    --model_name_or_path <pt_model_path_or_base_model> \
    --train_data_path <sft_single_turn_data_path> \
    --output_dir <sft_single_turn_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

**多GPU训练 (Accelerate)**:
```bash
accelerate launch --config_file accelerate_config.yaml qwen_sft_single_turn.py \
    --model_name_or_path <pt_model_path_or_base_model> \
    --train_data_path <sft_single_turn_data_path> \
    --output_dir <sft_single_turn_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### 多轮对话微调

针对需要联系上下文的多轮对话场景进行微调。

**单GPU训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python qwen_sft.py \
    --model_name_or_path <pt_model_path_or_base_model> \
    --train_data_path <sft_multi_turn_data_path> \
    --output_dir <sft_multi_turn_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

**多GPU训练 (Accelerate)**:
```bash
accelerate launch --config_file accelerate_config.yaml qwen_sft.py \
    --model_name_or_path <pt_model_path_or_base_model> \
    --train_data_path <sft_multi_turn_data_path> \
    --output_dir <sft_multi_turn_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### 🎯 考点
SFT这部分主要做是数据的格式，这个要强烈结合项目文档，疯狂改prompt，尝试不同格式的数据(这里展示了单轮对合和多轮对话格式的数据，其实还有其它类型)，这一步锻炼的也主要是工程能力，训完模型后评测结果，然后针对bad case进行纠正(比如说打负例，随机替换)等方式，还要对数据进行进一步纠错。当然，还有一些训练的trick，比如数据使用上，可以看cherry LLM(https://github.com/tianyi-lab/Cherry_LLM)这个项目 还有一些比如lora q-lora等技巧 这个一定要结合项目 lora的参数配置 lora3B vs qlora7B lora微调效果和全参微调效果的差异真的不到1%？ 这个都要进行考虑

### 强化学习对齐

通过强化学习方法，根据人类偏好或其他奖励信号进一步优化模型行为。

#### DPO (Direct Preference Optimization)

**单GPU训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python qwen_dpo.py \
    --model_name_or_path <sft_model_path> \
    --preference_data_path <dpo_preference_data_path> \
    --output_dir <dpo_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

**多GPU训练 (Accelerate)**:
```bash
accelerate launch --config_file accelerate_config.yaml qwen_dpo.py \
    --model_name_or_path <sft_model_path> \
    --preference_data_path <dpo_preference_data_path> \
    --output_dir <dpo_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### 🎯 考点与 DPO 详解

DPO (Direct Preference Optimization) 作为一种对齐语言模型与人类偏好的方法，与传统的基于强化学习的 RLHF 中的 PPO 等算法不同，它不需要显式地训练一个奖励模型，而是直接通过偏好数据来优化语言模型。

**DPO 损失函数**:

对于一个给定的偏好数据对 $(x, y_w, y_l)$，其中：
- $x$ 是输入的提示 (prompt)
- $y_w$ 是被选中的、更受偏好的回应 (winner response)
- $y_l$ 是被拒绝的、不太受偏好的回应 (loser response)

DPO的损失函数 $L_{DPO}$ 定义如下：
$L_{DPO}(\pi_{\theta}, \pi_{ref}) = -\log \sigma \left( \beta \left( \log \pi_{\theta}(y_w | x) - \log \pi_{ref}(y_w | x) \right) - \beta \left( \log \pi_{\theta}(y_l | x) - \log \pi_{ref}(y_l | x) \right) \right)$

我们可以将其中核心部分拆开来看：
令 $r_\theta(x, y) = \log \pi_\theta(y|x) - \log \pi_{ref}(y|x)$。这个 $r_\theta(x, y)$ 可以被理解为**策略模型 $\pi_\theta$ 相对于参考模型 $\pi_{ref}$ 为回应 $y$ 打出的"隐式奖励"的对数概率差**。

那么，损失函数可以简化为：
$L_{DPO} = -\log \sigma (\beta \cdot [r_\theta(x, y_w) - r_\theta(x, y_l)])$
其中 $\sigma$ 是 Sigmoid 函数。

**参数解释**:

1. **$\pi_\theta$ (策略模型 / Policy Model)**:
   - 这是我们正在训练和优化的语言模型。它的参数用 $\theta$ 表示。
   - 目标是调整 $\theta$ 使得这个模型更倾向于生成 $y_w$ 而不是 $y_l$。

2. **$\pi_{ref}$ (参考模型 / Reference Model)**:
   - 这通常是一个预训练好的、固定的SFT (Supervised Fine-Tuning) 模型。在DPO训练过程中，它的参数保持不变 (frozen)。
   - 它的作用是提供一个基准，防止策略模型 $\pi_\theta$ 为了满足偏好而偏离太远，从而导致语言质量下降或"模式崩溃"。

3. **$\log \pi_\theta(y|x)$**:
   - 表示在给定提示 $x$ 的条件下，策略模型 $\pi_\theta$ 生成回应 $y$ 的对数概率。这是通过对回应 $y$ 中所有token的条件对数概率求和得到的。
   - 计算 $\log \pi_\theta(y_w|x)$ 需要一次策略模型的前向传播 (Forward Pass 1)。
   - 计算 $\log \pi_\theta(y_l|x)$ 需要另一次策略模型的前向传播 (Forward Pass 3)。

4. **$\log \pi_{ref}(y|x)$**:
   - 表示在给定提示 $x$ 的条件下，参考模型 $\pi_{ref}$ 生成回应 $y$ 的对数概率。
   - 计算 $\log \pi_{ref}(y_w|x)$ 需要一次参考模型的前向传播 (Forward Pass 2)。
   - 计算 $\log \pi_{ref}(y_l|x)$ 需要另一次参考模型的前向传播 (Forward Pass 4)。
   - 这就是每个训练步骤**至少有4次模型前向传播**。

5. **$\beta$ (Beta / 温度系数)**:
   - 这是一个非常重要的超参数，通常取值在 (0, 1) 之间，例如 0.1, 0.25, 0.5。
   - 它控制了DPO损失对参考模型的依赖程度，或者说，它控制了策略模型 $\pi_\theta$ 被允许偏离参考模型 $\pi_{ref}$ 的程度。
   - **较小的 $\beta$**: 意味着对偏离参考模型的惩罚更大。策略模型会更倾向于与参考模型相似的生成分布，这有助于保持生成文本的流畅性和多样性，但可能对偏好的学习不够充分。
   - **较大的 $\beta$**: 意味着模型会更积极地学习偏好，即更努力地提高 $y_w$ 的概率并降低 $y_l$ 的概率（相对于参考模型）。但这可能导致模型过分拟合偏好数据中的某些模式，而牺牲一定的生成质量或泛化能力。
   - $\beta$ 的选择需要通过实验来确定最佳值。

**DPO 训练的工程要点** 🛠️
- **数据准备是核心中的核心**:
  - 高质量偏好数据对 $(prompt, chosen\_response, rejected\_response)$。数据的质量直接决定了模型对齐的效果。
  - 数据来源：可以来自人工标注、模型（如GPT-4）打分排序、已有开源偏好数据集等。
  - 偏好一致性与清晰度：标注标准需要统一，偏好差异需要明确。如果 "chosen" 和 "rejected" 的差异很小或模糊不清，模型很难学习。
- **模型设置**:
  - 策略模型 (Policy Model / Active Model)：通常是SFT阶段产出的模型副本，是我们要训练和优化的模型。
  - 参考模型 (Reference Model / SFT Model)：通常就是SFT阶段产出的模型，在DPO训练过程中其参数保持冻结。它是计算隐式奖励的基准。
- **DPO的优势与挑战**:
  - DPO 不需要显式训练奖励模型，简化了 RLHF 的流程。
  - 对偏好数据的质量和数量要求较高。高质量且充足的偏好数据是DPO成功的关键。

**下面是我对DPO的理解(这个是真是自己悟的)** 🤔:

DPO (Direct Preference Optimization) 的核心目标是让策略模型 ($\pi_\theta$) 能够更好地区分好的回应 ($y_w$) 和坏的回应 ($y_l$)，也就是要增大 $\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$ 这个值。

但是，直接这样做有两个潜在问题：
1. 模型可能会为了满足偏好而变得和它最初的样子（SFT模型）相差太大，丢失一些好的特性。
2. 如果一个偏好对本身就很容易区分（比如SFT模型已经能很好地给出 $y_w$ 远高于 $y_l$ 的概率），我们不希望模型从这种"简单"的例子中学习到过强的信号，好像它取得了巨大进步一样。

所以，DPO引入了参考模型 ($\pi_{ref}$)。它实际上关注的是策略模型区分优劣的能力相对于参考模型提高了多少，即 $(\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)) - (\log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x))$。

#### KTO (Kahneman-Tversky Optimization)

**单GPU训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python qwen_kto.py \
    --model_name_or_path <sft_model_path> \
    --kto_data_path <kto_data_path> \
    --output_dir <kto_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

**多GPU训练 (Accelerate)**:
```bash
accelerate launch --config_file accelerate_config.yaml qwen_kto.py \
    --model_name_or_path <sft_model_path> \
    --kto_data_path <kto_data_path> \
    --output_dir <kto_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### 🎯 考点与 KTO 详解

KTO通过对"不受偏好的样本"（即负样本，可以理解为"损失"）和"受偏好的样本"（即正样本，可以理解为"收益"）赋予不同的权重，从而更精细地调整模型行为。这意味着我们可以根据具体的任务场景，来决定是更侧重于"惩罚坏的回答"，还是更侧重于"奖励好的回答"。：

- **损失加权** (`undesirable_weight`) 与 **收益加权** (`desirable_weight`)：
  通过给不受偏好样本（负样本）和受偏好样本（正样本）赋予不同的权重，KTO 允许我们针对具体任务场景调整"惩罚坏回答" 与 "奖励好回答" 的相对力度。
- **β (Beta) 超参数**：与 DPO 类似，也控制策略模型允许偏离参考模型的幅度；但在 KTO 中通常与损失-收益权重一起网格搜索。
- **损失形式** (`loss_type="kto"`)：本质上是对 DPO 损失的加权 Sigmoid 交叉熵形式，可写为:

  $L_{KTO} = - [ w^+ \cdot \log\sigma(\beta (r_\theta(x,y_w) - r_\theta(x,y_l))) + w^- \cdot \log(1-\sigma(\beta (r_\theta(x,y_w) - r_\theta(x,y_l)))) ]$

  其中 $w^+$=desirable_weight，$w^-$=undesirable_weight。

**KTO 训练的工程要点** 🛠️

1. **样本拆分**：一条 DPO 数据会被拆成两条 KTO 样本（正/负），因此显存占用与 DPO 基本一致，但训练步数翻倍，要相应调整 `per_device_train_batch_size` 或 `gradient_accumulation_steps`。

2. **权重调参**：
   - 推荐先固定 $β$ 与 DPO 保持一致（如 0.1），随后网格搜索 `desirable_weight` / `undesirable_weight`（如 1/1、2/1、1/2）。
   - **经验**：在法律问答场景下，`undesirable_weight>desirable_weight` 往往能显著减少"胡说八道"现象，但过大可能导致回答过于保守。

3. **参考模型 (ref_model)**：
   - KTOTrainer 允许 `ref_model=None` 并通过 `precompute_ref_log_probs=True` 预计算参考对数概率，节省多卡显存。
   - 若要在线计算，可复用 SFT 模型并冻结参数，与 DPO 一致。

4. **指标监控**：
   - 训练中可同时监控 `kto/loss`、`kto/accuracy`（正样本判别准确率）。

5. **常见坑**：
   - 偏好数据噪声对 KTO 更敏感，建议在数据清洗阶段剔除"差距过小"。
   - 如果发现 loss 震荡且 accuracy 长期 <=50%，基本可判定样本标签对错或权重设置不合理。

**我的一些 KTO 训练感悟** 🤔
- 当**DPO 继续提升空间有限**，难以进一步提升时，引入KTO，特别是通过设置更高的 `undesirable_weight`（加大对坏回答的惩罚），可以成为进一步降低模型生成危险或不当回答概率的有效手段。
- KTO 的效果对 **负样本多样性** 极为敏感，实践中我会使用 GPT-4 辅助生成"带毒"回答，再经过人工过滤增强负样本丰富度。
- 由于 KTO 的损失是双向加权，训到后期很容易出现 **模式崩塌**（即模型倾向于只输出那些最安全、最通用、但可能缺乏信息量和多样性的回复）。此时减小 `undesirable_weight` 并缩小学习率通常能稳住模型。

#### PPO (Proximal Policy Optimization)

**单GPU训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python qwen_ppo.py \
    --model_name_or_path <sft_model_path> \
    --reward_model_path <reward_model_for_ppo_path> \
    --ppo_prompt_data_path <ppo_prompt_data_path> \
    --output_dir <ppo_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

**多GPU训练 (Accelerate)**:
```bash
accelerate launch --config_file accelerate_config.yaml qwen_ppo.py \
    --model_name_or_path <sft_model_path> \
    --reward_model_path <reward_model_for_ppo_path> \
    --ppo_prompt_data_path <ppo_prompt_data_path> \
    --output_dir <ppo_output_model_path>
    # (根据实际脚本参数添加其他必要配置)
```

#### GRPO (Group Relative Policy Optimization)

**单GPU训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python qwen_grpo.py \
    --model_name_or_path <sft_model_path> \
    --grpo_data_path <grpo_data_path> \
    --output_dir <grpo_output_model_path>
    # (根据实际脚本参数添加其他必要配置, 确保已安装 requirements_grpo.txt 中的依赖)
```

**多GPU训练 (Accelerate)**:
```bash
accelerate launch --config_file accelerate_config.yaml qwen_grpo.py \
    --model_name_or_path <sft_model_path> \
    --grpo_data_path <grpo_data_path> \
    --output_dir <grpo_output_model_path>
    # (根据实际脚本参数添加其他必要配置, 确保已安装 requirements_grpo.txt 中的依赖)
```

## 📈 模型评估

使用标准数据集或其他特定于法律领域的评估基准来衡量模型性能。这里使用常用的MMLU数据集进行评估。

```bash
python qwen_eval.py \
    --checkpoint_path <path_to_your_trained_model_checkpoint> \
    --eval_data_path <path_to_evaluation_data_or_config>
    # (根据实际脚本参数添加其他必要配置, 例如: --task legal_mmlU)
```

> 💡 建议评估不同阶段的模型（PT, SFT, DPO等）以追踪性能变化。

## 💬 推理与聊天机器人

加载训练好的模型进行交互式聊天或API服务。

```bash
python qwen_chat.py \
    --checkpoint_path <path_to_your_final_model_checkpoint>
    # (根据实际脚本参数添加其他必要配置)
```


## 🔧 其他脚本

该部分用于放置项目中其他辅助脚本或特定任务脚本的说明，例如数据转换、模型导出等。

### 单GPU运行
```bash
CUDA_VISIBLE_DEVICES=0 python xxx.py --params ...
```

### 多GPU运行 (Accelerate)
```bash
accelerate launch --config_file accelerate_config.yaml xxx.py --params ...
```

---

## 📝 注意事项

> 请根据您的具体脚本名称、参数和数据路径替换上述命令中的占位符 (例如 `<base_model_path>`, `<your_training_script.py>` 等)。

## 📄 License

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：[your-email@example.com]

