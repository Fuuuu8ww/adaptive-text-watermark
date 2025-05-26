# evaluation_utils.py（B）

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

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

    if not text or not text.strip():
        return text

    try:
        # 适配 Vamsi/T5_Paraphrase_Paws 的输入格式
        input_text_for_model = f"paraphrase: {text} </s>"

        inputs = attack_tokenizer(
            input_text_for_model, 
            return_tensors="pt", 
            truncation=True, 
            # pad_to_max_length=True, # Vamsi的示例用了，但通常truncation足够，可以测试是否需要
            max_length=max_input_length_for_attack 
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask # Vamsi的示例也用了attention_mask

        if input_ids.size(1) == 0:
            return text 

        with torch.no_grad():
            outputs = attack_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, # 传递attention_mask
                max_length=max_output_length_for_attack,
                do_sample=do_sample_for_attack,
                top_k=top_k_for_attack,
                top_p=top_p_for_attack,
                early_stopping=early_stopping_for_attack,
                num_return_sequences=1 # 我们只需要一个释义结果用于评估
                # num_beams 参数在 do_sample=True 时通常不与 top_k/top_p 一起主要使用，
                # 如果 do_sample=False，则 num_beams 会生效。
                # Vamsi的示例没有明确用num_beams配合do_sample=True，这里暂时也先不加，
                # 或者如果需要beam search，则设置do_sample=False。
                # 为了与示例更接近，如果用采样，就不设置num_beams，或者设为1。
            )
        
        paraphrased_text = attack_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        if not paraphrased_text or not paraphrased_text.strip():
            return text 
            
        return paraphrased_text

    except Exception:
        return text


# --- 3. 加权和目标函数 ---
# <<< 改进点 NORM_FLOOR_1: 在归一化时加入抬高底部逻辑 >>>
def calculate_objective_score_weighted_sum(
    metrics_dict: dict, 
    weights: dict,
    normalization_stats_updater, 
    penalty_for_invalid_ppl: float = 10000.0, 
    detection_score_floor_for_cost: float = 0.0 
    ) -> float:

    ppl_relative = metrics_dict.get('avg_ppl_relative', float('inf')) 
    original_detection_score = metrics_dict.get('avg_orig_detect', detection_score_floor_for_cost)
    robustness_diff_score = metrics_dict.get('robustness_diff', detection_score_floor_for_cost - 1.0) 

    print(f"DEBUG (Objective): Raw Metrics In: ppl_rel={ppl_relative:.3f}, orig_det={original_detection_score:.3f}, rob_diff={robustness_diff_score:.3f}")

    cost_ppl_relative_raw: float
    if torch.isnan(torch.tensor(ppl_relative)) or ppl_relative == float('inf'): # 保持对nan的检查
        cost_ppl_relative_raw = penalty_for_invalid_ppl 
    elif ppl_relative < -0.9: 
        cost_ppl_relative_raw = penalty_for_invalid_ppl 
    else:
        cost_ppl_relative_raw = ppl_relative

    cost_original_detection_raw = 1.0 - original_detection_score
    cost_robustness_diff_raw = 1.0 - robustness_diff_score

    print(f"DEBUG (Objective): Raw Costs: ppl_rel_raw={cost_ppl_relative_raw:.3f}, orig_det_raw={cost_original_detection_raw:.3f}, rob_diff_raw={cost_robustness_diff_raw:.3f}")

    raw_costs_for_norm = {
        'ppl_relative': cost_ppl_relative_raw,
        'cost_orig_detect': cost_original_detection_raw,
        'cost_robust_diff': cost_robustness_diff_raw
    }

    current_norm_params = normalization_stats_updater(raw_costs_for_norm)
    print(f"DEBUG (Objective): Normalization Params Used: {current_norm_params}")

    normalized_costs = {}
    epsilon = 1e-8 
    MIN_OBS_FOR_DYNAMIC_NORM = 3 
    COST_FLOOR_FOR_NORMALIZED_VALUE = 0.01 # 定义抬高后的成本下限

    for cost_type, raw_cost in raw_costs_for_norm.items():
        base_norm_val_for_floor_logic = 0.5
        if raw_cost == penalty_for_invalid_ppl or raw_cost == float('inf') or np.isnan(raw_cost): # 增加对nan的检查
            normalized_costs[cost_type] = penalty_for_invalid_ppl # 对于无效成本，直接使用惩罚值
            print(f"DEBUG (Objective Norm): For {cost_type}, invalid raw_cost ({raw_cost}), using penalty.")
            continue

        params = current_norm_params.get(cost_type)
        if params is None: 
            print(f"WARNING (Objective Norm): No norm params for {cost_type}. Using base_norm_val {base_norm_val_for_floor_logic} for floor. Raw: {raw_cost:.3f}")
        else:
            min_val, max_val = params['min'], params['max']
            if abs(max_val - min_val) < epsilon:
                # min 和 max 几乎相同。
                # 此时 base_norm_val_for_floor_logic 仍然是 0.5 (默认值)
                # 或者可以更细致地判断 raw_cost 相对于这个极小范围的位置，但0.5作为中性值是合理的
                print(f"DEBUG (Objective Norm Base): For {cost_type}, min approx max ({min_val:.3f} ~ {max_val:.3f}). Raw: {raw_cost:.3f}. Using base_norm_val: {base_norm_val_for_floor_logic:.3f} for floor logic.")
            else: # 窗口内有差异
                clipped_cost = np.clip(raw_cost, min_val, max_val)
                base_norm_val_for_floor_logic = (clipped_cost - min_val) / (max_val - min_val)
                print(f"DEBUG (Objective Norm Base): For {cost_type}, regular norm. Raw: {raw_cost:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}. Base_norm: {base_norm_val_for_floor_logic:.3f}")

        # 对所有有效的 base_norm_val_for_floor_logic (范围 [0,1] 或设定的0.5) 应用 "抬高底部"
        normalized_costs[cost_type] = COST_FLOOR_FOR_NORMALIZED_VALUE + base_norm_val_for_floor_logic * (1.0 - COST_FLOOR_FOR_NORMALIZED_VALUE)
        print(f"DEBUG (Objective Norm Raised): For {cost_type}, final raised norm_cost: {normalized_costs[cost_type]:.3f} (from base: {base_norm_val_for_floor_logic:.3f})")
    
    print(f"DEBUG (Objective): Normalized Costs: {normalized_costs}")

    cost_ppl_final = normalized_costs.get('ppl_relative', penalty_for_invalid_ppl)
    cost_original_detection_final = normalized_costs.get('cost_orig_detect', penalty_for_invalid_ppl) 
    cost_robustness_diff_final = normalized_costs.get('cost_robust_diff', penalty_for_invalid_ppl)

    # 确保如果原始成本是惩罚值，最终成本也是惩罚值
    if cost_ppl_relative_raw == penalty_for_invalid_ppl:
        cost_ppl_final = penalty_for_invalid_ppl
    if raw_costs_for_norm['cost_orig_detect'] == penalty_for_invalid_ppl: # 假设这种情况不会发生，除非PPL惩罚导致整个trial无效
        cost_original_detection_final = penalty_for_invalid_ppl
    if raw_costs_for_norm['cost_robust_diff'] == penalty_for_invalid_ppl:
        cost_robustness_diff_final = penalty_for_invalid_ppl
        
    print(f"DEBUG (Objective): Final Costs for Summation: ppl={cost_ppl_final:.3f}, orig_det={cost_original_detection_final:.3f}, rob_diff={cost_robustness_diff_final:.3f}")

    # 如果任何一个最终成本是惩罚值，总分也应该是惩罚值（或一个非常大的值）
    # 以防止其他项的“好”表现掩盖了一个致命缺陷
    if cost_ppl_final == penalty_for_invalid_ppl or \
       cost_original_detection_final == penalty_for_invalid_ppl or \
       cost_robustness_diff_final == penalty_for_invalid_ppl:
        print(f"DEBUG (Objective): At least one cost is penalty. Returning high score.")
        # 返回一个比任何可能的加权和都大的值，或者就是penalty_for_invalid_ppl（如果权重都为1）
        return penalty_for_invalid_ppl * sum(weights.values()) # 确保它足够大

    total_objective_score = (weights.get('w_ppl_relative', 1.0) * cost_ppl_final +
                             weights.get('w_original_detection', 1.0) * cost_original_detection_final +
                             weights.get('w_robustness_diff', 1.0) * cost_robustness_diff_final)
    
    print(f"DEBUG (Objective): Actual Total Objective Score (high precision): {total_objective_score:.10e}")
    return total_objective_score