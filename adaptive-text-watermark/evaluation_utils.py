# evaluation_utils.py（A）

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np # 确保导入 numpy


###修改###
##########
#PPL相对化
def calculate_relative_ppl_cost(
    ppl_score_watermarked: float,
    ppl_score_baseline: float,
    penalty_for_invalid_input: float = 10.0 # 当输入无效或基线PPL为0时，给予一个较大的惩罚值
                                             # 这个值的大小应相对于其他成本项的尺度
) -> float:
    """
    计算加水印文本PPL相对于无水印基线文本PPL的相对成本。
    成本 = (PPL_watermarked / PPL_baseline) - 1
    目标是使这个成本接近0（即水印不显著增加PPL）。
    如果水印反而降低了PPL，成本会被截断为0。
    """
    is_watermarked_invalid = np.isnan(ppl_score_watermarked) or \
                             np.isinf(ppl_score_watermarked) # 使用 np.isnan 和 np.isinf
    is_baseline_invalid = np.isnan(ppl_score_baseline) or \
                          np.isinf(ppl_score_baseline) # 使用 np.isnan 和 np.isinf

    if is_watermarked_invalid or is_baseline_invalid:
        return penalty_for_invalid_input

    if ppl_score_baseline <= 1e-6: # PPL 必须是正数，且不应非常接近0
        return penalty_for_invalid_input

    relative_ppl_increase = (ppl_score_watermarked / ppl_score_baseline) - 1.0
    cost_ppl_relative = max(0.0, relative_ppl_increase) # 成本不为负

    return cost_ppl_relative

# 新的加权和目标函数，使用相对PPL+非线性检测分数+鲁棒性差异
# calculate_objective_score_weighted_sum (保持与optimizer_basic.py中的名称一致)
def calculate_objective_score_weighted_sum(
    metrics_dict: dict, # 修改为接收一个包含所有平均指标的字典
    weights: dict,      # 修改为接收一个包含所有权重的字典
    normalization_stats_updater: callable, # 用于获取当前归一化统计数据 (min/max)
    penalty_for_invalid_ppl: float = 10.0, # 这个参数现在在 calculate_relative_ppl_cost 中处理
                                              # 但可以保留以防其他地方使用或作为配置传递
    detection_score_floor_for_cost: float = 0.0 # 当检测分数无效时使用的基准值 (用于原始成本计算)
) -> float:
    """
    计算基于加权和的目标函数得分（适配 optimizer_basic.py 的新结构）。
    包含: 相对PPL成本, 原始检测成本, 鲁棒性差异成本。
    成本项会进行归一化处理。
    目标是最小化这个得分。
    """

    # --- 1. 从 metrics_dict 提取平均指标 ---
    avg_ppl_relative = metrics_dict.get('avg_ppl_relative', float('inf'))
    avg_orig_detect = metrics_dict.get('avg_orig_detect', detection_score_floor_for_cost)
    # avg_attack_detect = metrics_dict.get('avg_attack_detect', detection_score_floor_for_cost) # 不直接用
    # avg_unwatermarked_detect = metrics_dict.get('avg_unwatermarked_detect', detection_score_floor_for_cost) # 不直接用
    robustness_diff = metrics_dict.get('robustness_diff', detection_score_floor_for_cost - 1.0) # 期望 robust_diff 尽可能大 (接近1)

    # --- 2. 计算原始成本项 (Raw Costs) ---
    # 这些是未归一化的成本，目标都是越小越好

    # a. 相对PPL成本 (已由 optimizer_basic 计算并传入 avg_ppl_relative)
    #    avg_ppl_relative 本身就是成本 (0 表示最好，越大越差)
    #    如果 avg_ppl_relative 是 inf，则使用 penalty_for_invalid_ppl (虽然上游已经处理过)
    cost_ppl_relative_raw: float
    if np.isinf(avg_ppl_relative) or np.isnan(avg_ppl_relative):
        cost_ppl_relative_raw = penalty_for_invalid_ppl # 应该与 calculate_relative_ppl_cost 中的 penalty_for_invalid_input 一致或相关
    else:
        cost_ppl_relative_raw = avg_ppl_relative

    # b. 原始检测成本
    #    avg_orig_detect 范围 [0, 1]，越高越好。
    #    成本 = 1.0 - avg_orig_detect。范围 [0, 1]，越小越好。
    #    如果 avg_orig_detect 无效 (例如 np.nan), cost 应该是一个较大的惩罚值。
    cost_orig_detect_raw: float
    if np.isnan(avg_orig_detect) or np.isinf(avg_orig_detect): # 检测分数不应是inf
        cost_orig_detect_raw = 1.0 - detection_score_floor_for_cost # 最大成本
    else:
        cost_orig_detect_raw = 1.0 - avg_orig_detect


    # c. 鲁棒性差异成本
    #    robustness_diff = avg_attack_detect - avg_unwatermarked_detect
    #    理想情况下 avg_attack_detect 接近1, avg_unwatermarked_detect 接近0 (或随机基线, e.g., 0.5)
    #    所以 robustness_diff 的理想值是 1 (或 0.5 如果基线是0.5)。目标是最大化此差异。
    #    成本 = Target_Robustness_Diff - robustness_diff。
    #    假设 Target_Robustness_Diff 为 1.0 (即希望攻击后检测率比无水印检测率高1个单位)。
    #    如果 robustness_diff 是 1.0，成本是 0。
    #    如果 robustness_diff 是 0.0，成本是 1.0。
    #    如果 robustness_diff 是 -1.0 (攻击后检测为0，无水印检测为1)，成本是 2.0。
    #    所以成本范围大致是 [0, 2] （如果以1为目标差异），越小越好。
    target_robustness_diff = 1.0 # 或者可以设为 (target_attack_detect - target_unwatermarked_detect)
                                  # 例如 target_attack_detect=1.0, target_unwatermarked_detect=0.0 -> target_robustness_diff=1.0
                                  # 如果无水印检测的理想基线是0.5，则 target_robustness_diff 可以是 1.0 - 0.5 = 0.5
    cost_robust_diff_raw: float
    if np.isnan(robustness_diff) or np.isinf(robustness_diff):
        cost_robust_diff_raw = target_robustness_diff - (detection_score_floor_for_cost - 1.0) # 一个较大的惩罚成本
    else:
        cost_robust_diff_raw = target_robustness_diff - robustness_diff


    # --- 3. 获取当前归一化统计数据并进行归一化 ---
    raw_costs_for_norm_update = {
        'ppl_relative': cost_ppl_relative_raw,
        'cost_orig_detect': cost_orig_detect_raw,
        'cost_robust_diff': cost_robust_diff_raw
    }
    current_norm_stats = normalization_stats_updater(raw_costs_for_norm_update)

    def normalize_cost(cost_value, cost_type_key, norm_stats_dict, min_valid_obs_for_norm=3):
        stats = norm_stats_dict.get(cost_type_key)
        if stats and stats.get('num_obs_in_window', 0) >= min_valid_obs_for_norm : # 确保有足够观测值才归一化
            min_val = stats['min']
            max_val = stats['max']
            if max_val > min_val: # 避免除以零
                # 将成本值裁剪到 [min_val, max_val] 范围内再归一化，避免极端值影响过大
                # clipped_cost_value = max(min_val, min(max_val, cost_value)) # 暂时不裁剪，直接用原始值
                # return (clipped_cost_value - min_val) / (max_val - min_val)
                # 对于 inf 或 nan 的原始成本，归一化后应该是一个表示“最差”的值，例如1.0或更大
                if np.isinf(cost_value) or np.isnan(cost_value):
                    return 1.5 # 或者一个比1.0大的固定惩罚值
                normalized = (cost_value - min_val) / (max_val - min_val)
                return max(0.0, min(1.0, normalized)) # 裁剪到0-1，防止数值问题
            elif max_val == min_val and max_val != float('inf') and max_val != float('-inf'): # 如果所有观测值都一样
                return 0.0 if cost_value == min_val else 1.0 # 如果当前值也一样，成本为0，否则为1（表示异常）
        # 如果统计数据不足或无效，则不进行归一化，或返回一个表示“中等惩罚”的值
        # 或者直接返回原始成本（如果原始成本尺度已经比较合理）
        # 为了简单，如果无法归一化，我们返回一个相对较大的值，除非它是0
        if np.isinf(cost_value) or np.isnan(cost_value):
            return 1.5 # 再次检查，以防原始成本就是inf/nan
        return cost_value if cost_value == 0 else max(1.0, cost_value) # 如果不能归一化，非零成本至少为1

    cost_ppl_normalized = normalize_cost(cost_ppl_relative_raw, 'ppl_relative', current_norm_stats)
    cost_orig_detect_normalized = normalize_cost(cost_orig_detect_raw, 'cost_orig_detect', current_norm_stats)
    cost_robust_diff_normalized = normalize_cost(cost_robust_diff_raw, 'cost_robust_diff', current_norm_stats)

    # --- 4. 计算加权和 (使用归一化后的成本) ---
    total_objective_score = (
        weights.get('w_ppl_relative', 1.0) * cost_ppl_normalized +
        weights.get('w_original_detection', 1.0) * cost_orig_detect_normalized +
        weights.get('w_robustness_diff', 1.0) * cost_robust_diff_normalized
    )

    # --- 调试信息 (可选) ---
    print(f"  DEBUG (Objective Advanced):")
    print(f"    Raw Costs: PPL_rel={cost_ppl_relative_raw:.3f}, OrigDet={cost_orig_detect_raw:.3f}, RobustDiff={cost_robust_diff_raw:.3f}")
    print(f"    Norm Stats (PPL_rel): min={current_norm_stats.get('ppl_relative', {}).get('min', 'N/A')}, max={current_norm_stats.get('ppl_relative', {}).get('max', 'N/A')}, obs={current_norm_stats.get('ppl_relative', {}).get('num_obs_in_window', 'N/A')}")
    print(f"    Norm Stats (OrigDet): min={current_norm_stats.get('cost_orig_detect', {}).get('min', 'N/A')}, max={current_norm_stats.get('cost_orig_detect', {}).get('max', 'N/A')}, obs={current_norm_stats.get('cost_orig_detect', {}).get('num_obs_in_window', 'N/A')}")
    print(f"    Norm Stats (RobustDiff): min={current_norm_stats.get('cost_robust_diff', {}).get('min', 'N/A')}, max={current_norm_stats.get('cost_robust_diff', {}).get('max', 'N/A')}, obs={current_norm_stats.get('cost_robust_diff', {}).get('num_obs_in_window', 'N/A')}")
    print(f"    Normalized Costs: PPL_norm={cost_ppl_normalized:.3f}, OrigDet_norm={cost_orig_detect_normalized:.3f}, RobustDiff_norm={cost_robust_diff_normalized:.3f}")
    print(f"    Weights: w_ppl={weights.get('w_ppl_relative', 1.0)}, w_orig_det={weights.get('w_original_detection', 1.0)}, w_rob_diff={weights.get('w_robustness_diff', 1.0)}")
    print(f"    Total Objective Score = {total_objective_score:.4f}")
    # --- ---

    return total_objective_score

#######over

# --- 1. 困惑度 (Perplexity) 计算函数 ---
def calculate_ppl(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_model_length: int = 512 # 应与传入tokenizer/model的最大长度匹配或较小
    ) -> float:
    """
    计算给定文本的困惑度 (Perplexity)。
    使用传入的model和tokenizer进行计算。
    """
    if not text or not text.strip():
        return float('inf')

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_model_length
        ).to(device)

        input_ids = inputs.input_ids

        if input_ids.size(1) <= 1:
            return float('inf')

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids) # labels=input_ids 用于计算PPL的loss
            loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            return float('inf')

        return torch.exp(loss).item()
    except Exception as e:
        # print(f"DEBUG (calculate_ppl): Exception occurred: {e} for text: '{text[:50]}...'") # 可选调试
        return float('inf')


# --- 2. 本地释义攻击模型函数 (适配 Vamsi/T5_Paraphrase_Paws) ---
def apply_local_paraphrase_attack(
    text: str,
    attack_model: AutoModelForSeq2SeqLM | None,
    attack_tokenizer: AutoTokenizer | None,
    device: torch.device,
    max_input_length_for_attack: int = 256,
    num_beams_for_attack: int = 3, # 这个参数在do_sample=True时通常不直接起作用
    max_output_length_for_attack: int = 256, # 与Vamsi/T5_Paraphrase_Paws示例一致
    # 以下参数根据 Vamsi/T5_Paraphrase_Paws 示例调整
    do_sample_for_attack: bool = True,
    top_k_for_attack: int = 120,
    top_p_for_attack: float = 0.95,
    early_stopping_for_attack: bool = True
    ) -> str:
    """
    使用本地部署的 Vamsi/T5_Paraphrase_Paws 模型进行释义攻击。
    输入格式为 "paraphrase: {sentence} </s>".
    如果攻击模型未提供或攻击失败，则返回原始文本。
    """
    if not attack_model or not attack_tokenizer:
        return text

    if not text or not text.strip(): # 增加对空文本或仅含空白字符文本的检查
        return text

    try:
        # 适配 Vamsi/T5_Paraphrase_Paws 的输入格式
        input_text_for_model = f"paraphrase: {text} </s>"

        inputs = attack_tokenizer(
            input_text_for_model,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length_for_attack
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        if input_ids.size(1) == 0: # 如果分词后为空
            return text

        # 生成参数
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_length": max_output_length_for_attack,
            "early_stopping": early_stopping_for_attack,
            "num_return_sequences": 1,
        }
        if do_sample_for_attack:
            generation_kwargs["do_sample"] = True
            generation_kwargs["top_k"] = top_k_for_attack
            generation_kwargs["top_p"] = top_p_for_attack
        else: # 如果不用采样，则可以使用beam search
            generation_kwargs["num_beams"] = num_beams_for_attack


        with torch.no_grad():
            outputs = attack_model.generate(**generation_kwargs)

        paraphrased_text = attack_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if not paraphrased_text or not paraphrased_text.strip(): # 如果释义结果为空
            return text

        return paraphrased_text

    except Exception as e:
        # print(f"DEBUG (apply_local_paraphrase_attack): Exception occurred: {e} for text: '{text[:50]}...'") # 可选调试
        return text