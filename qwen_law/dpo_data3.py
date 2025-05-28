import os
import torch # 仍然导入，但 API LLM 不直接使用。如果其他部分不需要，可以移除。
import pandas as pd # 仅用于创建 DataFrame 后导出，如果直接写 JSONL，可以考虑移除或仅在调试时使用。
from tqdm.auto import tqdm
# from datasets import load_dataset # 在此版本中不用于 JSONL 加载
import json # 用于加载和写入 JSONL

from distilabel.llms import OpenAILLM
# from distilabel.llms import TransformersLLM # 不再使用本地 TransformersLLM 进行生成
from distilabel.steps.tasks import TextGeneration, EvolQuality, UltraFeedback
# from tqdm import tqdm # tqdm.auto 通常是更好的选择
# from utils.utils import find_files # 在此版本中未使用

# --- 配置信息 ---
TMP_PATH = "/hy-tmp/dpo_data" # 数据集缓存目录 (如果 EvolQuality/UltraFeedback 内部使用)
OUTPUT_DATA_PATH = "/root/llm_post_train/qwen/data" # 最终输出文件的目录
INPUT_JSONL_FILE = '/root/llm_post_train/qwen/data/generated_sft_dataset.jsonl' # <--- 重要：请设置你的输入 JSONL 文件路径

# 生成模型 (模型 A 和 模型 B) 的 API 配置
# 请替换为你的实际 API 密钥和在 SiliconFlow 上的正确模型 ID
MODEL_SCOPE_API_KEY = "sk-kqxmywozwdltfnyommzmniuzergcsgcgmyxrbehxlzdjetnh" # <--- 重要：请设置你的 SILICONFLOW API 密钥
MODEL_SCOPE_API_KEY_B = "sk-yzbvynfwlcmvcydvvnflckjehoocmfeshawoefkihtojmvjf"
MODEL_SCOPE_BASE_URL = "https://api.siliconflow.cn/v1"

MODEL_A_ID_SILICONFLOW = "Qwen/Qwen3-32B" # 示例，请使用你在 SiliconFlow 上的模型 A 的正确 ID
MODEL_B_ID_SILICONFLOW = "THUDM/GLM-Z1-32B-0414" # <--- 重要：请验证并设置在 SiliconFlow 上的模型 B 的正确 ID

# 评估模型 (DeepSeek) 的 API 配置
DS_API_KEY = "sk-pvhtfraiykmbipqtexjcslnbugcalhkfolkahdqiehibgfxx" # <--- 重要：请设置你的 DEEPSEEK API 密钥
DS_BASE_URL = "https://api.siliconflow.cn/v1"
DS_MODEL = "deepseek-ai/DeepSeek-R1"

TRAIN_SUBSET = -1 # 设置为 0 或 -1 以处理所有数据
os.makedirs(TMP_PATH, exist_ok=True)
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

# --- 从 JSONL 加载数据 ---
def load_questions_from_jsonl(file_path, subset_size):
    data_entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 合并 instruction 和 input 以形成 "question"
            question = f"{data.get('instruction', '')}\n\n{data.get('input', '')}".strip()
            data_entries.append({"question": question, "original_data": data}) # 如果需要，保留原始数据
    
    if subset_size > 0 and len(data_entries) > subset_size:
        return [entry["question"] for entry in data_entries[:subset_size]]
    return [entry["question"] for entry in data_entries]

questions = load_questions_from_jsonl(INPUT_JSONL_FILE, TRAIN_SUBSET)

if not questions:
    print(f"未能从 {INPUT_JSONL_FILE} 加载任何问题。请检查文件和路径。")
    exit()

print(f"已加载 {len(questions)} 个问题。")

# --- 初始化 LLMs ---

# 生成模型 A
if not MODEL_SCOPE_API_KEY or MODEL_SCOPE_API_KEY == 'YOUR_MODEL_SCOPE_API_KEY_HERE':
    print("错误：请为生成模型设置有效的 MODEL_SCOPE_API_KEY。")
    exit()

llm_a = OpenAILLM(
    model=MODEL_A_ID_SILICONFLOW,
    base_url=MODEL_SCOPE_BASE_URL,
    api_key=MODEL_SCOPE_API_KEY
)
llm_a.load() # 通常，基于 API 的 LLM 没有像本地模型那样繁重的 load() 操作，但 distilabel 可能会使用它。

# 生成模型 B
llm_b = OpenAILLM(
    model=MODEL_B_ID_SILICONFLOW,
    base_url=MODEL_SCOPE_BASE_URL,
    api_key=MODEL_SCOPE_API_KEY_B
)
llm_b.load()

# 评估模型
if not DS_API_KEY or DS_API_KEY == 'YOUR_DEEPSEEK_API_KEY_HERE':
    print("错误：请为评估模型设置有效的 DS_API_KEY。")
    exit()

judge_llm = OpenAILLM(
    model=DS_MODEL,
    base_url=DS_BASE_URL,
    api_key=DS_API_KEY,
    generation_kwargs={
        "max_new_tokens": 8000,  # 修复：使用 max_new_tokens 而不是 max_tokens
        "temperature": 0.7
    }
)
judge_llm.load()

# --- 初始化 Distilabel 步骤 ---
evol_quality = EvolQuality(llm=judge_llm, num_evolutions=1)
evol_quality.load()

ultrafeedback = UltraFeedback(llm=judge_llm)
ultrafeedback.load()

records = []

def chat(msg):
    return [{"role": "user", "content": msg}]

# 添加重试函数
def safe_process_with_retry(processor, input_data, max_retries=3, process_name="Unknown"):
    """安全地处理数据，带重试机制"""
    for attempt in range(max_retries):
        try:
            print(f"    尝试 {process_name} (第 {attempt + 1} 次)...")
            processed = processor.process(input_data)
            result = next(processed, None)
            if result:
                print(f"    {process_name} 成功")
                return result
            else:
                print(f"    {process_name} 返回空结果")
        except Exception as e:
            print(f"    {process_name} 第 {attempt + 1} 次尝试失败: {e}")
            if attempt == max_retries - 1:
                print(f"    {process_name} 所有重试都失败了")
                return None
    return None

GEN_KWARGS = dict(max_new_tokens=4000, temperature=0.7) # 减少token数量避免超时

for q_idx, q_text in enumerate(tqdm(questions, desc="生成/改写/评分")):
    print(f"\n正在处理问题 {q_idx+1}/{len(questions)}: {q_text[:100]}...") # 打印问题开头
    current_question_data = { # 用于记录当前问题的所有中间和最终数据
        "question": q_text,
        "original_response_j": "处理中...",
        "original_response_k": "处理中...",
        "response_j": "处理中...",
        "response_k": "处理中...",
        "rating_j": -1.0,
        "rating_k": -1.0,
        "rationales_j": "N/A",
        "rationales_k": "N/A",
        "error_stage": "None",
        "error_message": "None"
    }

    try:
        # (i) 两个模型独立生成
        print("  正在从模型 A 生成响应...")
        gen_a_outputs = llm_a.generate(inputs=[chat(q_text)], **GEN_KWARGS)
        print(f"  原始模型 A 输出 (gen_a_outputs): 类型={type(gen_a_outputs)}, 值={str(gen_a_outputs)[:200]}") # 调试信息

        # --- 模型 A 响应处理 ---
        gen_a = "错误: 模型 A 未能生成有效响应。" # 默认值
        if gen_a_outputs and isinstance(gen_a_outputs, list) and len(gen_a_outputs) > 0:
            first_output_a = gen_a_outputs[0]
            print(f"  模型 A 第一个输出元素 (first_output_a): 类型={type(first_output_a)}, 值={str(first_output_a)[:200]}") # 调试信息
            if isinstance(first_output_a, dict):
                generations_a = first_output_a.get("generations")
                print(f"  模型 A generations_a: 类型={type(generations_a)}, 值={str(generations_a)[:200]}") # 调试信息
                if isinstance(generations_a, list) and len(generations_a) > 0:
                    first_generation_a = generations_a[0]
                    print(f"  模型 A first_generation_a: 类型={type(first_generation_a)}, 值={str(first_generation_a)[:200]}") # 调试信息
                    if isinstance(first_generation_a, dict):
                        content_a = first_generation_a.get("content")
                        if content_a is not None:
                            gen_a = content_a
                        else:
                            gen_a = "错误: 模型 A 响应中缺少 'content' 键。"
                            print(f"  警告: {gen_a} 详细: {first_generation_a}")
                    elif isinstance(first_generation_a, str): # 新增的逻辑：如果直接是字符串
                        gen_a = first_generation_a # 直接使用这个字符串作为生成结果
                        print(f"  信息: 模型 A 的 'generation' 直接是字符串内容，已获取。")
                    else:
                        gen_a = f"错误: 模型 A 的 'generation' 不是字典类型，而是 {type(first_generation_a)}。"
                        print(f"  警告: {gen_a} 详细: {first_generation_a}")
                else:
                    gen_a = "错误: 模型 A 的 'generations' 列表为空或无效。"
                    print(f"  警告: {gen_a} 详细: {generations_a}")
            else:
                gen_a = f"错误: 模型 A 的输出元素不是字典类型，而是 {type(first_output_a)}。"
                print(f"  警告: {gen_a} 详细: {first_output_a}")
        else:
            gen_a = "错误: 模型 A 的 generate() 调用返回了空或无效的输出。"
            print(f"  警告: {gen_a} 详细: {gen_a_outputs}")
        current_question_data["original_response_j"] = gen_a
        print(f"  模型 A 解析后响应 (前50字符): {gen_a[:50]}")

        # --- 模型 B 响应处理 (与模型 A 类似) ---
        print("  正在从模型 B 生成响应...")
        gen_b_outputs = llm_b.generate(inputs=[chat(q_text)], **GEN_KWARGS)
        print(f"  原始模型 B 输出 (gen_b_outputs): 类型={type(gen_b_outputs)}, 值={str(gen_b_outputs)[:200]}") # 调试信息

        gen_b = "错误: 模型 B 未能生成有效响应。" # 默认值
        if gen_b_outputs and isinstance(gen_b_outputs, list) and len(gen_b_outputs) > 0:
            first_output_b = gen_b_outputs[0]
            print(f"  模型 B 第一个输出元素 (first_output_b): 类型={type(first_output_b)}, 值={str(first_output_b)[:200]}")
            if isinstance(first_output_b, dict):
                generations_b = first_output_b.get("generations")
                print(f"  模型 B generations_b: 类型={type(generations_b)}, 值={str(generations_b)[:200]}")
                if isinstance(generations_b, list) and len(generations_b) > 0:
                    first_generation_b = generations_b[0]
                    print(f"  模型 B first_generation_b: 类型={type(first_generation_b)}, 值={str(first_generation_b)[:200]}")
                    if isinstance(first_generation_b, dict):
                        content_b = first_generation_b.get("content")
                        if content_b is not None:
                            gen_b = content_b
                        else:
                            gen_b = "错误: 模型 B 响应中缺少 'content' 键。"
                            print(f"  警告: {gen_b} 详细: {first_generation_b}")
                    elif isinstance(first_generation_b, str): # 新增的逻辑：如果直接是字符串
                        gen_b = first_generation_b # 直接使用这个字符串作为生成结果
                        print(f"  信息: 模型 B 的 'generation' 直接是字符串内容，已获取。")
                    else:
                        gen_b = f"错误: 模型 B 的 'generation' 不是字典类型，而是 {type(first_generation_b)}。"
                        print(f"  警告: {gen_b} 详细: {first_generation_b}")
                else:
                    gen_b = "错误: 模型 B 的 'generations' 列表为空或无效。"
                    print(f"  警告: {gen_b} 详细: {generations_b}")
            else:
                gen_b = f"错误: 模型 B 的输出元素不是字典类型，而是 {type(first_output_b)}。"
                print(f"  警告: {gen_b} 详细: {first_output_b}")
        else:
            gen_b = "错误: 模型 B 的 generate() 调用返回了空或无效的输出。"
            print(f"  警告: {gen_b} 详细: {gen_b_outputs}")
        current_question_data["original_response_k"] = gen_b
        print(f"  模型 B 解析后响应 (前50字符): {gen_b[:50]}")

        # (ii) EvolQuality 迭代 1 次
        # --- EvolQuality 模型 A ---
        print("  正在进化模型 A 的响应...")
        evo_a = gen_a  # 默认使用原始响应
        if isinstance(gen_a, str) and not gen_a.startswith("错误:"): # 仅当 gen_a 是有效字符串时尝试进化
            evo_a_input = [{"instruction": q_text, "response": gen_a}]
            evo_a_result = safe_process_with_retry(evol_quality, evo_a_input, process_name="EvolQuality A")
            
            if evo_a_result and isinstance(evo_a_result, list) and len(evo_a_result) > 0:
                first_evo_a = evo_a_result[0]
                if isinstance(first_evo_a, dict):
                    evolved_response_a = first_evo_a.get("evolved_response")
                    if evolved_response_a and isinstance(evolved_response_a, str):
                        evo_a = evolved_response_a
                        print(f"  模型 A 进化成功")
                    else:
                        print(f"  警告: EvolQuality A 响应中缺少有效的 'evolved_response'")
                else:
                    print(f"  警告: EvolQuality A 的结果元素不是字典类型")
            else:
                print(f"  警告: EvolQuality A 处理失败，使用原始响应")
        else:
            print("  跳过模型 A 进化，因为原始响应有错误")
        current_question_data["response_j"] = evo_a
        print(f"  模型 A 进化后的响应 (前50字符): {evo_a[:50]}")

        # --- EvolQuality 模型 B ---
        print("  正在进化模型 B 的响应...")
        evo_b = gen_b  # 默认使用原始响应
        if isinstance(gen_b, str) and not gen_b.startswith("错误:"):
            evo_b_input = [{"instruction": q_text, "response": gen_b}]
            evo_b_result = safe_process_with_retry(evol_quality, evo_b_input, process_name="EvolQuality B")
            
            if evo_b_result and isinstance(evo_b_result, list) and len(evo_b_result) > 0:
                first_evo_b = evo_b_result[0]
                if isinstance(first_evo_b, dict):
                    evolved_response_b = first_evo_b.get("evolved_response")
                    if evolved_response_b and isinstance(evolved_response_b, str):
                        evo_b = evolved_response_b
                        print(f"  模型 B 进化成功")
                    else:
                        print(f"  警告: EvolQuality B 响应中缺少有效的 'evolved_response'")
                else:
                    print(f"  警告: EvolQuality B 的结果元素不是字典类型")
            else:
                print(f"  警告: EvolQuality B 处理失败，使用原始响应")
        else:
            print("  跳过模型 B 进化，因为原始响应有错误")
        current_question_data["response_k"] = evo_b
        print(f"  模型 B 进化后的响应 (前50字符): {evo_b[:50]}")

        # (iii) UltraFeedback 打分
        print("  正在使用 UltraFeedback 评分...")
        ratings = [3.0, 3.0] # 默认中等评分，而不是错误评分
        rationales = ["默认评分：响应质量中等", "默认评分：响应质量中等"]
        
        # 检查是否有有效的响应可以评分
        valid_evo_a = isinstance(evo_a, str) and not evo_a.startswith("错误:") and len(evo_a.strip()) > 10
        valid_evo_b = isinstance(evo_b, str) and not evo_b.startswith("错误:") and len(evo_b.strip()) > 10
        
        if valid_evo_a and valid_evo_b:
            feedback_input = [{"instruction": q_text, "generations": [evo_a, evo_b]}]
            feedback_result = safe_process_with_retry(ultrafeedback, feedback_input, process_name="UltraFeedback")
            
            if feedback_result and isinstance(feedback_result, list) and len(feedback_result) > 0:
                first_feedback = feedback_result[0]
                if isinstance(first_feedback, dict):
                    ratings_val = first_feedback.get("ratings")
                    rationales_val = first_feedback.get("rationales")
                    
                    # 处理评分
                    if isinstance(ratings_val, list) and len(ratings_val) >= 2:
                        # 安全转换评分，处理None值和无效值
                        safe_ratings = []
                        for i, rating in enumerate(ratings_val[:2]):
                            if rating is not None:
                                try:
                                    # 确保评分在1-5范围内
                                    safe_rating = max(1.0, min(5.0, float(rating)))
                                    safe_ratings.append(safe_rating)
                                except (ValueError, TypeError):
                                    print(f"  警告: 评分 {i} 无法转换为数字: {rating}")
                                    safe_ratings.append(3.0)  # 默认中等评分
                            else:
                                print(f"  警告: 评分 {i} 为 None")
                                safe_ratings.append(3.0)  # 默认中等评分
                        
                        if len(safe_ratings) >= 2:
                            ratings = safe_ratings[:2]
                            print(f"  UltraFeedback 评分成功: {ratings}")
                        else:
                            print(f"  警告: 有效评分数量不足，使用默认评分")
                    else:
                        print(f"  警告: UltraFeedback 返回的 ratings 格式不正确: {ratings_val}")
                        # 使用简单的长度比较作为备用评分
                        if len(evo_a) > len(evo_b):
                            ratings = [4.0, 3.0]
                            rationales = ["备用评分：响应A更详细", "备用评分：响应B较简短"]
                        elif len(evo_b) > len(evo_a):
                            ratings = [3.0, 4.0]
                            rationales = ["备用评分：响应A较简短", "备用评分：响应B更详细"]
                        else:
                            ratings = [3.5, 3.5]
                            rationales = ["备用评分：两个响应长度相似", "备用评分：两个响应长度相似"]
                    
                    # 处理评分理由
                    if isinstance(rationales_val, list) and len(rationales_val) >= 2:
                        rationales = [str(r) if r is not None else "未提供理由" for r in rationales_val[:2]]
                    elif isinstance(rationales_val, list) and len(rationales_val) == 1:
                        rationales = [str(rationales_val[0]) if rationales_val[0] is not None else "未提供理由", "未提供详细理由"]
                    else:
                        print(f"  警告: UltraFeedback 返回的 rationales 格式不正确: {rationales_val}")
                        # 保持之前设置的备用理由
                else:
                    print(f"  警告: UltraFeedback 的结果不是字典格式")
                    # 使用备用评分方案
                    ratings = [3.5, 3.5]
                    rationales = ["评分失败：使用默认评分", "评分失败：使用默认评分"]
            else:
                print(f"  警告: UltraFeedback 处理失败，使用备用评分方案")
                # 基于响应质量的简单评分
                if valid_evo_a and not valid_evo_b:
                    ratings = [4.0, 2.0]
                    rationales = ["响应A有效", "响应B无效或过短"]
                elif valid_evo_b and not valid_evo_a:
                    ratings = [2.0, 4.0]
                    rationales = ["响应A无效或过短", "响应B有效"]
                else:
                    ratings = [3.5, 3.5]
                    rationales = ["备用评分：两个响应都有效", "备用评分：两个响应都有效"]
        else:
            print("  跳过 UltraFeedback 评分，因为响应质量不足")
            if not valid_evo_a and not valid_evo_b:
                ratings = [1.0, 1.0]
                rationales = ["响应A无效或过短", "响应B无效或过短"]
            elif not valid_evo_a:
                ratings = [1.0, 3.0]
                rationales = ["响应A无效或过短", "响应B有效"]
            else:  # not valid_evo_b
                ratings = [3.0, 1.0]
                rationales = ["响应A有效", "响应B无效或过短"]

        current_question_data["rating_j"] = ratings[0]
        current_question_data["rating_k"] = ratings[1]
        current_question_data["rationales_j"] = rationales[0]
        current_question_data["rationales_k"] = rationales[1]
        print(f"  最终评分: 模型 A: {current_question_data['rating_j']}, 模型 B: {current_question_data['rating_k']}")
        
        # 添加获胜者信息
        if ratings[0] > ratings[1]:
            current_question_data["winner"] = "response_j"
            print(f"  获胜者: 模型 A (评分差: {ratings[0] - ratings[1]:.1f})")
        elif ratings[1] > ratings[0]:
            current_question_data["winner"] = "response_k"
            print(f"  获胜者: 模型 B (评分差: {ratings[1] - ratings[0]:.1f})")
        else:
            current_question_data["winner"] = "tie"
            print(f"  结果: 平局")

    except Exception as e:
        print(f"处理问题 {q_idx+1} 的主 try 块时发生严重错误: {q_text[:100]}. 错误: {e}")
        current_question_data["error_stage"] = "MainLoop"
        current_question_data["error_message"] = str(e)
        # 确保即使发生意外错误，错误信息也会被记录
        if current_question_data["original_response_j"] == "处理中...": current_question_data["original_response_j"] = "处理错误"
        if current_question_data["original_response_k"] == "处理中...": current_question_data["original_response_k"] = "处理错误"
        if current_question_data["response_j"] == "处理中...": current_question_data["response_j"] = "处理错误"
        if current_question_data["response_k"] == "处理中...": current_question_data["response_k"] = "处理错误"

    records.append(current_question_data) # 将当前问题的所有数据（包括可能的错误信息）添加到记录中
    if torch.cuda.is_available(): # 仅在CUDA可用时尝试清空缓存
        torch.cuda.empty_cache() # 如果任何底层库使用了 GPU，最好清空缓存。

# ... (脚本的其余部分，如保存结果，保持不变)

# --- 保存结果为 JSONL 文件 ---
# 添加统计信息
print("\n" + "="*50)
print("处理结果统计:")
print("="*50)

total_questions = len(records)
successful_generations_a = sum(1 for r in records if not r["original_response_j"].startswith("错误:"))
successful_generations_b = sum(1 for r in records if not r["original_response_k"].startswith("错误:"))
successful_evolutions_a = sum(1 for r in records if not r["response_j"].startswith("错误:") and r["response_j"] != r["original_response_j"])
successful_evolutions_b = sum(1 for r in records if not r["response_k"].startswith("错误:") and r["response_k"] != r["original_response_k"])
valid_ratings = sum(1 for r in records if r["rating_j"] > 0 and r["rating_k"] > 0)

print(f"总问题数: {total_questions}")
print(f"模型A生成成功率: {successful_generations_a}/{total_questions} ({successful_generations_a/total_questions*100:.1f}%)")
print(f"模型B生成成功率: {successful_generations_b}/{total_questions} ({successful_generations_b/total_questions*100:.1f}%)")
print(f"模型A进化成功数: {successful_evolutions_a}")
print(f"模型B进化成功数: {successful_evolutions_b}")
print(f"有效评分数: {valid_ratings}/{total_questions} ({valid_ratings/total_questions*100:.1f}%)")

if valid_ratings > 0:
    avg_rating_a = sum(r["rating_j"] for r in records if r["rating_j"] > 0) / valid_ratings
    avg_rating_b = sum(r["rating_k"] for r in records if r["rating_k"] > 0) / valid_ratings
    wins_a = sum(1 for r in records if r.get("winner") == "response_j")
    wins_b = sum(1 for r in records if r.get("winner") == "response_k")
    ties = sum(1 for r in records if r.get("winner") == "tie")
    
    print(f"模型A平均评分: {avg_rating_a:.2f}")
    print(f"模型B平均评分: {avg_rating_b:.2f}")
    print(f"获胜统计: A获胜={wins_a}, B获胜={wins_b}, 平局={ties}")

print("="*50)

output_file = os.path.join(OUTPUT_DATA_PATH, "dpo_data.jsonl") # 修改输出文件后缀为 .jsonl
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False) # ensure_ascii=False 以正确处理中文字符
            f.write(json_line + '\n')
    print(f"\n处理完成。数据已保存到 {output_file}")
except IOError as e:
    print(f"保存文件时出错: {output_file}. 错误: {e}")
except Exception as e:
    print(f"保存结果时发生未知错误: {e}")