# evaluation_utils_vC.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

# --- 1. 困惑度 (Perplexity) 计算函数 ---
def calculate_ppl(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_model_length: int = 512
    ) -> float:
    """
    计算给定文本的困惑度 (Perplexity)。
    使用传入的model和tokenizer进行计算。
    """
    if not text or not text.strip():
        return float('inf')
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_model_length).to(device)
        input_ids = inputs.input_ids
        if input_ids.size(1) <= 1:
            return float('inf')
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float('inf')
        return torch.exp(loss).item()
    except Exception:
        return float('inf')


# --- 2. 本地释义攻击模型函数 (适配 Vamsi/T5_Paraphrase_Paws) ---
def apply_local_paraphrase_attack(
    text: str,
    attack_model: AutoModelForSeq2SeqLM | None,
    attack_tokenizer: AutoTokenizer | None,
    device: torch.device,
    max_input_length_for_attack: int = 256,
    num_beams_for_attack: int = 3,
    max_output_length_for_attack: int = 256,
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
    if not text or not text.strip():
        return text
    try:
        input_text_for_model = f"paraphrase: {text} </s>"
        inputs = attack_tokenizer(input_text_for_model, return_tensors="pt", truncation=True, max_length=max_input_length_for_attack).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        if input_ids.size(1) == 0:
            return text
        generation_kwargs = {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "max_length": max_output_length_for_attack, "early_stopping": early_stopping_for_attack,
            "num_return_sequences": 1,
        }
        if do_sample_for_attack:
            generation_kwargs["do_sample"] = True
            generation_kwargs["top_k"] = top_k_for_attack
            generation_kwargs["top_p"] = top_p_for_attack
        else:
            generation_kwargs["num_beams"] = num_beams_for_attack
        with torch.no_grad():
            outputs = attack_model.generate(**generation_kwargs)
        paraphrased_text = attack_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not paraphrased_text or not paraphrased_text.strip():
            return text
        return paraphrased_text
    except Exception:
        return text


# --- 3. 版本 C: 加权和目标函数 (主要用于记录一个分数，实际选择基于原始指标) ---
def calculate_objective_score_paper_centric(
    metrics_dict: dict,
    weights: dict,
    penalty_for_invalid_metric: float = 10000.0,
    detection_score_floor: float = 0.0,
) -> float:
    """
    版本 C: 计算目标函数得分，更直接地组合论文中关注的指标。
    在网格搜索中，这个函数的返回值主要用于记录，人工选择将基于原始指标。
    目标是最小化这个得分（如果用于优化器的话）。
    """

    # --- 1. 从 metrics_dict 提取平均指标 ---
    avg_ppl_watermarked = metrics_dict.get('avg_ppl_watermarked', float('inf'))
    avg_ppl_baseline = metrics_dict.get('avg_ppl_unwatermarked_baseline', float('inf'))
    avg_orig_detect = metrics_dict.get('avg_orig_detect', detection_score_floor)
    avg_attack_detect = metrics_dict.get('avg_attack_detect', detection_score_floor)

    # --- 2. 计算各项成本 (目标是使成本越小越好) ---
    cost_ppl: float
    if np.isinf(avg_ppl_watermarked) or np.isnan(avg_ppl_watermarked) or \
       np.isinf(avg_ppl_baseline) or np.isnan(avg_ppl_baseline) or avg_ppl_baseline <= 1e-6:
        cost_ppl = penalty_for_invalid_metric
    else:
        relative_increase = (avg_ppl_watermarked / avg_ppl_baseline) - 1.0
        cost_ppl = max(0.0, relative_increase)

    cost_orig_detect: float
    if np.isnan(avg_orig_detect) or np.isinf(avg_orig_detect):
        cost_orig_detect = 1.0 - detection_score_floor
    else:
        cost_orig_detect = 1.0 - avg_orig_detect

    cost_attack_detect: float
    if np.isnan(avg_attack_detect) or np.isinf(avg_attack_detect):
        cost_attack_detect = 1.0 - detection_score_floor
    else:
        cost_attack_detect = 1.0 - avg_attack_detect

    # --- 3. 处理无效总成本 ---
    current_costs = [cost_ppl, cost_orig_detect, cost_attack_detect]
    if penalty_for_invalid_metric in current_costs:
        # 对于网格搜索记录，我们仍然计算一个值，但人工分析时会注意到原始的inf/nan
        # 或者直接返回一个非常大的值，让其在“目标分数”排序中靠后
        calculated_score = penalty_for_invalid_metric * sum(weights.values())
        if calculated_score == 0 and penalty_for_invalid_metric > 0 : # 如果权重都是0，避免返回0
             calculated_score = penalty_for_invalid_metric
        print(f"  DEBUG (Objective Paper-Centric): At least one raw metric invalid, assigning high score: {calculated_score}")
        return calculated_score


    # --- 4. 计算加权和 ---
    total_objective_score = (
        weights.get('w_ppl', 1.0) * cost_ppl +
        weights.get('w_orig_detect', 1.0) * cost_orig_detect +
        weights.get('w_attack_detect', 1.0) * cost_attack_detect
    )

    # --- 调试信息 ---
    print(f"  DEBUG (Objective Paper-Centric):")
    print(f"    Metrics In: PPL_wm={avg_ppl_watermarked:.2f}, PPL_base={avg_ppl_baseline:.2f}, OrigDet={avg_orig_detect:.3f}, AttackDet={avg_attack_detect:.3f}")
    print(f"    Costs: PPL={cost_ppl:.3f}, OrigDet={cost_orig_detect:.3f}, AttackDet={cost_attack_detect:.3f}")
    print(f"    Weights: w_ppl={weights.get('w_ppl',1.0)}, w_orig_det={weights.get('w_orig_detect',1.0)}, w_att_det={weights.get('w_attack_detect',1.0)}")
    print(f"    Calculated Objective Score = {total_objective_score:.4f}")

    return total_objective_score