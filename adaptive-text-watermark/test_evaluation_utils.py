# test_evaluation_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import numpy as np # 确保导入

# 从你的 evaluation_utils.py 文件中导入函数
from evaluation_utils import (
    calculate_ppl,
    apply_local_paraphrase_attack,
    calculate_relative_ppl_cost,
    calculate_objective_score_weighted_sum # 注意，这个函数签名和你给的有点不一样，我按照你的最新版本来
)

# --- 全局测试配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Test using device: {DEVICE} ---")

# --- 辅助函数和模型加载 (轻量级) ---
def get_small_causal_lm():
    model_name = "distilgpt2" # 一个非常小的因果语言模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: # distilgpt2 可能没有 pad_token
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        model.eval()
        print(f"Successfully loaded small Causal LM: {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load small Causal LM ({model_name}): {e}")
        return None, None

def get_small_seq2seq_lm():
    model_name = "t5-small" # 一个非常小的序列到序列模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        model.eval()
        print(f"Successfully loaded small Seq2Seq LM: {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load small Seq2Seq LM ({model_name}): {e}")
        return None, None

# --- 测试用例 ---

def test_calculate_ppl_function():
    print("\n--- Testing calculate_ppl ---")
    model, tokenizer = get_small_causal_lm()
    if model is None or tokenizer is None:
        print("Skipping calculate_ppl test due to model loading failure.")
        return

    test_text_valid = "This is a test sentence for perplexity."
    ppl_valid = calculate_ppl(test_text_valid, model, tokenizer, DEVICE, max_model_length=128)
    print(f"PPL for '{test_text_valid}': {ppl_valid}")
    assert isinstance(ppl_valid, float) and ppl_valid > 0, "PPL for valid text failed."

    test_text_empty = ""
    ppl_empty = calculate_ppl(test_text_empty, model, tokenizer, DEVICE)
    print(f"PPL for empty text: {ppl_empty}")
    assert ppl_empty == float('inf'), "PPL for empty text should be inf."

    test_text_short = "A" # 单个token，或分词后少于等于1个token
    ppl_short = calculate_ppl(test_text_short, model, tokenizer, DEVICE)
    print(f"PPL for short text '{test_text_short}': {ppl_short}")
    assert ppl_short == float('inf'), "PPL for very short text should be inf."
    print("calculate_ppl tests passed (basic checks).")

def test_apply_local_paraphrase_attack_function():
    print("\n--- Testing apply_local_paraphrase_attack ---")
    attack_model, attack_tokenizer = get_small_seq2seq_lm()
    if attack_model is None or attack_tokenizer is None:
        print("Skipping apply_local_paraphrase_attack test due to model loading failure.")
        return

    test_text = "This is a simple sentence to be paraphrased."
    paraphrased_text = apply_local_paraphrase_attack(
        test_text, attack_model, attack_tokenizer, DEVICE,
        max_input_length_for_attack=64, max_output_length_for_attack=64
    )
    print(f"Original: '{test_text}'")
    print(f"Paraphrased: '{paraphrased_text}'")
    assert isinstance(paraphrased_text, str), "Paraphrase output type failed."
    if paraphrased_text != test_text: # t5-small可能只是简单复述或略微改变
        print("Paraphrase seems to have changed the text (as expected with a model).")
    else:
        print("Paraphrase returned original text (might happen with small model or if attack is ineffective).")

    # 测试无模型情况
    paraphrased_no_model = apply_local_paraphrase_attack(test_text, None, None, DEVICE)
    assert paraphrased_no_model == test_text, "Paraphrase with no model should return original text."
    print("apply_local_paraphrase_attack tests passed (basic checks).")

def test_calculate_relative_ppl_cost_function():
    print("\n--- Testing calculate_relative_ppl_cost ---")
    cost1 = calculate_relative_ppl_cost(ppl_score_watermarked=10.0, ppl_score_baseline=5.0)
    print(f"Relative PPL cost (10.0 vs 5.0): {cost1}") # (10/5)-1 = 1.0
    assert abs(cost1 - 1.0) < 1e-6, "Relative PPL cost calculation error."

    cost2 = calculate_relative_ppl_cost(ppl_score_watermarked=4.0, ppl_score_baseline=5.0)
    print(f"Relative PPL cost (4.0 vs 5.0, should be 0): {cost2}") # (4/5)-1 = -0.2, max(0, -0.2) = 0
    assert abs(cost2 - 0.0) < 1e-6, "Relative PPL cost (watermarked < baseline) error."

    penalty_val = 10.0
    cost_inf_wm = calculate_relative_ppl_cost(ppl_score_watermarked=float('inf'), ppl_score_baseline=5.0, penalty_for_invalid_input=penalty_val)
    print(f"Relative PPL cost (inf wm vs 5.0): {cost_inf_wm}")
    assert cost_inf_wm == penalty_val, "Relative PPL cost with inf watermarked PPL failed."

    cost_zero_base = calculate_relative_ppl_cost(ppl_score_watermarked=10.0, ppl_score_baseline=0.0, penalty_for_invalid_input=penalty_val)
    print(f"Relative PPL cost (10.0 vs 0.0 base): {cost_zero_base}")
    assert cost_zero_base == penalty_val, "Relative PPL cost with zero baseline PPL failed."
    print("calculate_relative_ppl_cost tests passed.")

def test_calculate_objective_score_weighted_sum_function():
    print("\n--- Testing calculate_objective_score_weighted_sum ---")

    # 模拟一个 normalization_stats_updater
    # 在真实场景中，这个 updater 会维护和更新 min/max 统计
    # 为了测试，我们让它返回一些固定的或可预测的 min/max
    def mock_normalization_stats_updater(raw_costs):
        print(f"    Mock updater received raw_costs: {raw_costs}")
        # 假设对于这个测试，我们不进行有效的归一化，或者说min/max范围导致归一化结果等于原始成本
        # 或者提供一个固定的min/max
        mock_stats = {}
        for key in raw_costs.keys():
            mock_stats[key] = {'min': 0.0, 'max': 2.0, 'num_obs_in_window': 10} # 示例范围
            # 如果希望测试不归一化的情况（例如统计不足）
            # mock_stats[key] = {'min': float('inf'), 'max': float('-inf'), 'num_obs_in_window': 0}
        print(f"    Mock updater returning mock_stats: {mock_stats}")
        return mock_stats

    # 1. 测试理想情况 (所有指标都很好)
    metrics_ideal = {
        'avg_ppl_relative': 0.01,       # PPL几乎没有增加 (成本接近0)
        'avg_orig_detect': 0.99,        # 原始检测很好 (成本接近0)
        'robustness_diff': 0.95         # 攻击后检测 - 无水印检测 差异很大 (成本接近0, 假设target_diff=1)
                                        # (0.99 - 0.04 = 0.95)
    }
    weights_default = {
        'w_ppl_relative': 1.0,
        'w_original_detection': 1.0,
        'w_robustness_diff': 1.0
    }
    score_ideal = calculate_objective_score_weighted_sum(
        metrics_dict=metrics_ideal,
        weights=weights_default,
        normalization_stats_updater=mock_normalization_stats_updater,
        penalty_for_invalid_ppl=20.0, # 这里的penalty值应与calculate_relative_ppl_cost中的一致
        detection_score_floor_for_cost=0.0
    )
    print(f"Objective score (ideal metrics): {score_ideal}")
    # 期望分数较低。具体值取决于mock_normalization_stats_updater的行为和target_robustness_diff
    # 如果 mock_normalization_stats_updater 返回的min=0, max=2，且成本都落在这个范围内:
    # cost_ppl_raw = 0.01 -> norm = (0.01-0)/2 = 0.005
    # cost_orig_raw = 1-0.99 = 0.01 -> norm = (0.01-0)/2 = 0.005
    # cost_robust_raw = 1-0.95 = 0.05 -> norm = (0.05-0)/2 = 0.025
    # total = 1*0.005 + 1*0.005 + 1*0.025 = 0.035
    assert score_ideal < 0.1, "Score for ideal metrics seems too high." # 这是一个粗略的断言

    # 2. 测试PPL差的情况
    metrics_bad_ppl = {
        'avg_ppl_relative': 5.0,        # PPL增加了5倍 (成本5.0)
        'avg_orig_detect': 0.9,
        'robustness_diff': 0.8
    }
    score_bad_ppl = calculate_objective_score_weighted_sum(
        metrics_dict=metrics_bad_ppl,
        weights=weights_default,
        normalization_stats_updater=mock_normalization_stats_updater
    )
    print(f"Objective score (bad PPL): {score_bad_ppl}")
    # cost_ppl_raw = 5.0 -> norm = (5-0)/2 (如果裁剪) = 1.0, or (5-0)/2 = 2.5 (如果不裁剪)
    # 假设 mock_stats 的 max 足够大，或者 normalize_cost 做了裁剪
    # 如果都用上面的 min=0, max=2 归一化：
    # cost_ppl_raw = 5.0 -> norm (裁剪到max) = (2.0-0)/2 = 1.0 (因为5>2)
    # cost_orig_raw = 1-0.9 = 0.1 -> norm = (0.1-0)/2 = 0.05
    # cost_robust_raw = 1-0.8 = 0.2 -> norm = (0.2-0)/2 = 0.1
    # total = 1*1.0 + 1*0.05 + 1*0.1 = 1.15
    assert score_bad_ppl > score_ideal, "Score for bad PPL should be higher."

    # 3. 测试鲁棒性差的情况
    metrics_bad_robustness = {
        'avg_ppl_relative': 0.1,
        'avg_orig_detect': 0.7,        # 原始检测一般 (成本0.3)
        'robustness_diff': -0.5        # 攻击后检测远不如无水印检测 (成本1 - (-0.5) = 1.5)
    }
    score_bad_robustness = calculate_objective_score_weighted_sum(
        metrics_dict=metrics_bad_robustness,
        weights=weights_default,
        normalization_stats_updater=mock_normalization_stats_updater
    )
    print(f"Objective score (bad robustness): {score_bad_robustness}")
    # cost_ppl_raw = 0.1 -> norm = (0.1-0)/2 = 0.05
    # cost_orig_raw = 1-0.7 = 0.3 -> norm = (0.3-0)/2 = 0.15
    # cost_robust_raw = 1-(-0.5) = 1.5 -> norm = (1.5-0)/2 = 0.75
    # total = 1*0.05 + 1*0.15 + 1*0.75 = 0.95
    assert score_bad_robustness > score_ideal, "Score for bad robustness should be higher."
    print("calculate_objective_score_weighted_sum tests passed (basic checks).")


if __name__ == "__main__":
    print("Starting tests for evaluation_utils.py...")
    test_calculate_ppl_function()
    test_apply_local_paraphrase_attack_function()
    test_calculate_relative_ppl_cost_function()
    test_calculate_objective_score_weighted_sum_function() # 注意这个函数的签名可能需要对齐
    print("\nAll tests finished.")
    