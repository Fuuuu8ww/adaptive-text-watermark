# optimizer_basic_grid_search_vC.py

import os
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer as SentenceTransformerLib

from watermark import Watermark # 假设 watermark.py 在同一目录或 PYTHONPATH
from evaluation_utils_vC import ( # 导入版本 C 的 utils
    calculate_ppl,
    apply_local_paraphrase_attack,
    calculate_objective_score_paper_centric # 使用新的目标函数（主要用于记录）
)
from model import SemanticModel # 假设 model.py 在同一目录或 PYTHONPATH
try:
    from utils import vocabulary_mapping # 假设 utils.py 在同一目录或 PYTHONPATH
except ImportError:
    print("ERROR: Could not import 'vocabulary_mapping' from utils.py. Ensure it's available.")
    exit()


# --- 1. Configuration Section ---
print("--- Optimizer Configuration (GRID SEARCH DATA COLLECTION MODE) ---")

# Model Paths
BASE_MODEL_PATH = "/data2/szr/"
WATERMARK_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "opt-305m")
MEASURE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "gpt-large-2")
EMBEDDING_MODEL_NAME = os.path.join(BASE_MODEL_PATH, "sentence-transformers_all-mpnet-base-v2")
SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH = os.path.join(BASE_MODEL_PATH, "adaptive-text-watermark", "model", "semantic_mapping_model.pth")
LOCAL_ATTACK_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "adaptive-text-watermark", "T5-paragraph")
PPL_EVAL_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "Meta-Llama-3-8B-Instruct-hf")

print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print(f"WATERMARK_MODEL_PATH: {WATERMARK_MODEL_PATH}")
print(f"MEASURE_MODEL_PATH: {MEASURE_MODEL_PATH}")
print(f"EMBEDDING_MODEL_NAME (or path): {EMBEDDING_MODEL_NAME}")
print(f"SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH: {SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH}")
print(f"LOCAL_ATTACK_MODEL_PATH: {LOCAL_ATTACK_MODEL_PATH}")
print(f"PPL_EVAL_MODEL_PATH: {PPL_EVAL_MODEL_PATH}")


# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Grid Search Configuration ---
ALPHA_GRID = [1, 2, 3, 4]  # Example grid for alpha
DELTA_GRID = [0.5, 1.0, 1.5, 2.0] # Example grid for delta
# ALPHA_GRID = [2, 3]  # 更小的网格用于快速测试
# DELTA_GRID = [1.0, 1.5] # 更小的网格用于快速测试

print(f"Grid Search: Alpha values: {ALPHA_GRID}")
print(f"Grid Search: Delta values: {DELTA_GRID}")
TOTAL_GRID_POINTS = len(ALPHA_GRID) * len(DELTA_GRID)
print(f"Total grid points to evaluate: {TOTAL_GRID_POINTS}")

# Test Prompts File
TEST_PROMPTS_FILE = "test_prompts.txt"
NUM_TEST_PROMPTS_TO_USE = 5 # 可以减少提示数量以加速网格搜索
print(f"Test Prompts: Using '{TEST_PROMPTS_FILE}', evaluating first {NUM_TEST_PROMPTS_TO_USE} prompts.")

N_PPL_SAMPLES = 2 # 可以减少PPL采样次数以加速网格搜索
print(f"PPL Averaging: Using {N_PPL_SAMPLES} samples for PPL calculation.")

GENERATION_CONFIG = {
    'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.1, 'no_repeat_ngram_size': 0,
    'max_new_tokens': 100, 'min_new_tokens': 80, # 可以减少生成长度以加速
    'secret_string': 'The quick brown fox jumps over the lazy dog',
    'measure_threshold': 50, 'delta_0': 1.0
}

PPL_MAX_MODEL_LENGTH = 1024

ATTACK_MAX_INPUT_LENGTH = 128 # 配合生成长度调整
ATTACK_MAX_OUTPUT_LENGTH = 128
ATTACK_DO_SAMPLE = True
ATTACK_TOP_K = 120
ATTACK_TOP_P = 0.95
ATTACK_EARLY_STOPPING = True
ATTACK_NUM_BEAMS = 3

# 目标函数权重 (主要用于 calculate_objective_score_paper_centric 的记录)
OBJECTIVE_WEIGHTS_PAPER_CENTRIC = {
    'w_ppl': 1.0,
    'w_orig_detect': 1.0,
    'w_attack_detect': 1.0,
}
PENALTY_FOR_INVALID_METRIC_IN_OBJECTIVE = 10000.0
DETECTION_SCORE_FLOOR_IN_OBJECTIVE = 0.0
print(f"Objective Weights (Paper-Centric, for record keeping): {OBJECTIVE_WEIGHTS_PAPER_CENTRIC}")


# --- Global variables ---
LOADED_MODELS_CACHE = None
ALL_TEST_PROMPTS_CACHE = None
BASELINE_PPL_CACHE = {}
ALL_GRID_SEARCH_RESULTS = [] # 用于存储网格搜索的所有结果

# --- Model Loading Function (load_all_models_once) ---
def load_all_models_once():
    global LOADED_MODELS_CACHE, PPL_MAX_MODEL_LENGTH
    if LOADED_MODELS_CACHE is not None:
        print("Models already loaded from cache.")
        return LOADED_MODELS_CACHE
    models_dict = {}
    print(f"\n--- Loading All Models to {DEVICE} ---")
    try:
        print(f"Loading Watermark Model & Tokenizer from: {WATERMARK_MODEL_PATH} ...")
        models_dict['watermark_tokenizer'] = AutoTokenizer.from_pretrained(WATERMARK_MODEL_PATH)
        models_dict['watermark_model'] = AutoModelForCausalLM.from_pretrained(WATERMARK_MODEL_PATH).to(DEVICE)
        models_dict['watermark_model'].eval()
        try: # 尝试从水印模型获取初始PPL长度（可能被PPL评估模型覆盖）
            PPL_MAX_MODEL_LENGTH = models_dict['watermark_model'].config.max_position_embeddings
        except AttributeError: pass # 保持默认
        print(f"  Initial PPL_MAX_MODEL_LENGTH (from watermark model or default): {PPL_MAX_MODEL_LENGTH}")
        print("  Watermark Model & Tokenizer: OK")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Watermark Model/Tokenizer from '{WATERMARK_MODEL_PATH}': {e}")
        return None
    try:
        print(f"Loading Measurement Model & Tokenizer from: {MEASURE_MODEL_PATH} ...")
        models_dict['measure_tokenizer'] = AutoTokenizer.from_pretrained(MEASURE_MODEL_PATH)
        models_dict['measure_model'] = AutoModelForCausalLM.from_pretrained(MEASURE_MODEL_PATH).to(DEVICE)
        models_dict['measure_model'].eval()
        print("  Measurement Model & Tokenizer: OK")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Measurement Model/Tokenizer from '{MEASURE_MODEL_PATH}': {e}")
        return None
    try:
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME} ...")
        models_dict['embedding_model'] = SentenceTransformerLib(EMBEDDING_MODEL_NAME, device=DEVICE)
        print("  Embedding Model: OK")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Embedding Model '{EMBEDDING_MODEL_NAME}': {e}")
        return None
    try:
        print(f"Loading Semantic Mapping Model from: {SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH} ...")
        models_dict['transform_model'] = SemanticModel()
        models_dict['transform_model'].load_state_dict(torch.load(SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH, map_location=DEVICE))
        models_dict['transform_model'].to(DEVICE)
        models_dict['transform_model'].eval()
        print("  Semantic Mapping Model: OK")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Semantic Mapping Model from '{SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH}': {e}")
        return None
    models_dict['attack_tokenizer'] = None
    models_dict['attack_model'] = None
    if os.path.exists(LOCAL_ATTACK_MODEL_PATH):
        print(f"Loading Local Attack Model & Tokenizer from: {LOCAL_ATTACK_MODEL_PATH} ...")
        try:
            models_dict['attack_tokenizer'] = AutoTokenizer.from_pretrained(LOCAL_ATTACK_MODEL_PATH)
            models_dict['attack_model'] = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_ATTACK_MODEL_PATH).to(DEVICE)
            models_dict['attack_model'].eval()
            print("  Local Attack Model & Tokenizer: OK")
        except Exception as e:
            print(f"  WARNING: Failed to load Local Attack Model/Tokenizer from '{LOCAL_ATTACK_MODEL_PATH}': {e}. Paraphrase attack will be skipped/return original text.")
    else:
        print(f"WARNING: Local Attack Model path not found: '{LOCAL_ATTACK_MODEL_PATH}'. Paraphrase attack will be skipped/return original text.")
    try:
        print(f"Loading PPL Evaluation Model & Tokenizer from: {PPL_EVAL_MODEL_PATH} ...")
        models_dict['ppl_eval_tokenizer'] = AutoTokenizer.from_pretrained(PPL_EVAL_MODEL_PATH)
        models_dict['ppl_eval_model'] = AutoModelForCausalLM.from_pretrained(
            PPL_EVAL_MODEL_PATH,
            torch_dtype=torch.bfloat16, # 或者 torch.float16
            device_map="auto"
        ).eval()
        try:
            PPL_MAX_MODEL_LENGTH_NEW = models_dict['ppl_eval_model'].config.max_position_embeddings
            if hasattr(models_dict['ppl_eval_tokenizer'], 'model_max_length'):
                 PPL_MAX_MODEL_LENGTH_NEW = min(PPL_MAX_MODEL_LENGTH_NEW, models_dict['ppl_eval_tokenizer'].model_max_length)
            if PPL_MAX_MODEL_LENGTH_NEW != PPL_MAX_MODEL_LENGTH: # 只有当新值不同且更优时才更新
                 PPL_MAX_MODEL_LENGTH = PPL_MAX_MODEL_LENGTH_NEW
                 print(f"  PPL max length updated from PPL evaluation model config: {PPL_MAX_MODEL_LENGTH}")
            else:
                 print(f"  PPL max length remains: {PPL_MAX_MODEL_LENGTH}")
        except AttributeError:
            print(f"  Warning: Could not get max_position_embeddings from PPL evaluation model. Using PPL max length: {PPL_MAX_MODEL_LENGTH}")
        print("  PPL Evaluation Model & Tokenizer: OK")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load PPL Evaluation Model/Tokenizer from '{PPL_EVAL_MODEL_PATH}': {e}")
        return None
    try:
        if 'watermark_model' not in models_dict or not hasattr(models_dict['watermark_model'], 'config') or not hasattr(models_dict['watermark_model'].config, 'vocab_size'):
            print("FATAL ERROR: watermark_model or its config or vocab_size is not available for generating mapping_list.")
            return None
        vocab_size_for_mapping = models_dict['watermark_model'].config.vocab_size
        print(f"  Using watermark_model.config.vocab_size ({vocab_size_for_mapping}) for generating mapping_list.")
        models_dict['mapping_list'] = vocabulary_mapping(vocab_size_for_mapping, 384, seed=66)
        print(f"  Mapping list generated. Length: {len(models_dict['mapping_list'])} (should match {vocab_size_for_mapping}).")
    except Exception as e:
        print(f"FATAL ERROR: Failed to generate mapping list: {e}")
        return None
    models_dict['device'] = DEVICE
    print("--- All available models loaded successfully. ---")
    LOADED_MODELS_CACHE = models_dict
    return models_dict

# --- Baseline PPL Calculation Function ---
def get_or_calculate_baseline_ppl(prompt_text, watermark_obj_for_unwatermarked_gen, loaded_models):
    global BASELINE_PPL_CACHE, N_PPL_SAMPLES
    cache_key = prompt_text
    if cache_key in BASELINE_PPL_CACHE:
        return BASELINE_PPL_CACHE[cache_key]
    # print(f"    Calculating baseline PPL (avg over {N_PPL_SAMPLES} samples) for prompt: \"{prompt_text[:50]}...\"")
    generated_baseline_text_samples = []
    prompt_baseline_ppls = []
    for _ in range(N_PPL_SAMPLES):
        unwatermarked_text_sample = watermark_obj_for_unwatermarked_gen.generate_unwatermarked(prompt_text)
        generated_baseline_text_samples.append(unwatermarked_text_sample)
        if not unwatermarked_text_sample or not unwatermarked_text_sample.strip():
            ppl_sample = float('inf')
        else:
            ppl_sample = calculate_ppl(
                unwatermarked_text_sample,
                loaded_models['ppl_eval_model'], loaded_models['ppl_eval_tokenizer'],
                loaded_models['device'], PPL_MAX_MODEL_LENGTH
            )
        prompt_baseline_ppls.append(ppl_sample)
    valid_ppls = [p for p in prompt_baseline_ppls if p != float('inf') and not np.isnan(p)]
    avg_baseline_ppl_for_prompt = np.mean(valid_ppls) if valid_ppls else float('inf')
    representative_baseline_text = next((txt for txt in generated_baseline_text_samples if txt and txt.strip()), "")
    BASELINE_PPL_CACHE[cache_key] = (avg_baseline_ppl_for_prompt, representative_baseline_text)
    # print(f"      Average Baseline PPL: {avg_baseline_ppl_for_prompt:.2f}, Rep. Text: \"{representative_baseline_text[:30]}...\" (from {len(valid_ppls)} valid samples)")
    return avg_baseline_ppl_for_prompt, representative_baseline_text

# --- Evaluation Function (slightly simplified for grid search data collection) ---
def evaluate_single_params_for_grid(alpha_val, delta_val, current_loaded_models, current_test_prompts):
    trial_start_time = time.time()
    current_alpha = int(alpha_val)
    current_delta = float(delta_val)
    print(f"  Evaluating: alpha={current_alpha}, delta={current_delta:.1f}")

    # 默认返回值，以防实例化失败
    default_metrics_on_error = {
        'avg_ppl_watermarked': float('inf'), 'avg_ppl_unwatermarked_baseline': float('inf'),
        'avg_ppl_relative': float('inf'), 'avg_orig_detect': DETECTION_SCORE_FLOOR_IN_OBJECTIVE,
        'avg_attack_detect': DETECTION_SCORE_FLOOR_IN_OBJECTIVE,
        'avg_unwatermarked_detect': DETECTION_SCORE_FLOOR_IN_OBJECTIVE,
        'robustness_diff': DETECTION_SCORE_FLOOR_IN_OBJECTIVE - 1.0,
        'calculated_objective_score': PENALTY_FOR_INVALID_METRIC_IN_OBJECTIVE * sum(OBJECTIVE_WEIGHTS_PAPER_CENTRIC.values())
    }
    try:
        watermark_obj = Watermark(
            device=current_loaded_models['device'], watermark_tokenizer=current_loaded_models['watermark_tokenizer'],
            measure_tokenizer=current_loaded_models['measure_tokenizer'], watermark_model=current_loaded_models['watermark_model'],
            measure_model=current_loaded_models['measure_model'], embedding_model=current_loaded_models['embedding_model'],
            transform_model=current_loaded_models['transform_model'], mapping_list=current_loaded_models['mapping_list'],
            alpha=current_alpha, delta=current_delta, **GENERATION_CONFIG )
    except Exception as e:
        print(f"    ERROR: Failed to instantiate Watermark object: {e}")
        return default_metrics_on_error # 返回包含所有键的字典

    prompt_metrics_list = []
    prompts_to_run = current_test_prompts[:NUM_TEST_PROMPTS_TO_USE]

    for i, prompt_text in enumerate(prompts_to_run):
        current_prompt_metrics = {'ppl_watermarked': float('inf'), 'ppl_unwatermarked_baseline': float('inf'),
                                  'ppl_relative': float('inf'), 'original_detection': DETECTION_SCORE_FLOOR_IN_OBJECTIVE,
                                  'attacked_detection': DETECTION_SCORE_FLOOR_IN_OBJECTIVE,
                                  'unwatermarked_detection': DETECTION_SCORE_FLOOR_IN_OBJECTIVE}
        try:
            ppl_unwatermarked_baseline_avg, representative_baseline_text_for_detection = get_or_calculate_baseline_ppl(prompt_text, watermark_obj, current_loaded_models)
            current_prompt_metrics['ppl_unwatermarked_baseline'] = ppl_unwatermarked_baseline_avg
            generated_watermarked_text_samples = []
            for _ in range(N_PPL_SAMPLES): generated_watermarked_text_samples.append(watermark_obj.generate_adaptive_watermarke(prompt_text))
            current_prompt_watermarked_ppls = []
            for text_sample in generated_watermarked_text_samples:
                if not text_sample or not text_sample.strip(): current_prompt_watermarked_ppls.append(float('inf'))
                else: current_prompt_watermarked_ppls.append(calculate_ppl( text_sample, current_loaded_models['ppl_eval_model'],
                        current_loaded_models['ppl_eval_tokenizer'], current_loaded_models['device'], PPL_MAX_MODEL_LENGTH))
            valid_ppls_for_current_wm = [p for p in current_prompt_watermarked_ppls if p != float('inf') and not np.isnan(p)]
            avg_ppl_watermarked_for_prompt = np.mean(valid_ppls_for_current_wm) if valid_ppls_for_current_wm else float('inf')
            current_prompt_metrics['ppl_watermarked'] = avg_ppl_watermarked_for_prompt
            watermarked_text_for_downstream = next((txt for txt in generated_watermarked_text_samples if txt and txt.strip()), "")

            if ppl_unwatermarked_baseline_avg > 0 and ppl_unwatermarked_baseline_avg != float('inf') and avg_ppl_watermarked_for_prompt != float('inf'):
                current_prompt_metrics['ppl_relative'] = (avg_ppl_watermarked_for_prompt / ppl_unwatermarked_baseline_avg) - 1.0
            else: current_prompt_metrics['ppl_relative'] = float('inf')

            if not watermarked_text_for_downstream:
                print(f"    Skipping detection/attack for prompt {i+1} due to empty watermarked_text.")
                # unwatermarked_detection 仍然可以计算
                if representative_baseline_text_for_detection and representative_baseline_text_for_detection.strip():
                    unwatermarked_score = watermark_obj.detection(representative_baseline_text_for_detection)
                    current_prompt_metrics['unwatermarked_detection'] = unwatermarked_score if unwatermarked_score is not None else DETECTION_SCORE_FLOOR_IN_OBJECTIVE
                prompt_metrics_list.append(current_prompt_metrics)
                continue

            orig_score = watermark_obj.detection(watermarked_text_for_downstream)
            current_prompt_metrics['original_detection'] = orig_score if orig_score is not None else DETECTION_SCORE_FLOOR_IN_OBJECTIVE

            if representative_baseline_text_for_detection and representative_baseline_text_for_detection.strip():
                unwatermarked_score = watermark_obj.detection(representative_baseline_text_for_detection)
                current_prompt_metrics['unwatermarked_detection'] = unwatermarked_score if unwatermarked_score is not None else DETECTION_SCORE_FLOOR_IN_OBJECTIVE

            attacked_text = apply_local_paraphrase_attack( watermarked_text_for_downstream, current_loaded_models.get('attack_model'), current_loaded_models.get('attack_tokenizer'),
                current_loaded_models['device'], ATTACK_MAX_INPUT_LENGTH, ATTACK_NUM_BEAMS, ATTACK_MAX_OUTPUT_LENGTH, ATTACK_DO_SAMPLE, ATTACK_TOP_K, ATTACK_TOP_P, ATTACK_EARLY_STOPPING)
            attack_score = DETECTION_SCORE_FLOOR_IN_OBJECTIVE
            if attacked_text and attacked_text.strip() and attacked_text != watermarked_text_for_downstream:
                attack_score_val = watermark_obj.detection(attacked_text)
                if attack_score_val is not None: attack_score = attack_score_val
            current_prompt_metrics['attacked_detection'] = attack_score
            prompt_metrics_list.append(current_prompt_metrics)
        except Exception as e:
            print(f"    ERROR during prompt {i+1} evaluation: {e}")
            # 尝试从缓存获取基线PPL，即使当前prompt出错
            cached_baseline_data = BASELINE_PPL_CACHE.get(prompt_text)
            current_prompt_metrics['ppl_unwatermarked_baseline'] = cached_baseline_data[0] if cached_baseline_data else float('inf')
            prompt_metrics_list.append(current_prompt_metrics) # 添加包含部分或全部inf/default值的metrics

    def safe_mean(values, default_val):
        valid_values = [v for v in values if v is not None and v != float('inf') and not np.isnan(v)]
        return np.mean(valid_values) if valid_values else default_val

    aggregated_metrics = {}
    aggregated_metrics['avg_ppl_watermarked'] = safe_mean([pm.get('ppl_watermarked', float('inf')) for pm in prompt_metrics_list], float('inf'))
    aggregated_metrics['avg_ppl_unwatermarked_baseline'] = safe_mean([pm.get('ppl_unwatermarked_baseline', float('inf')) for pm in prompt_metrics_list], float('inf'))
    aggregated_metrics['avg_ppl_relative'] = safe_mean([pm.get('ppl_relative', float('inf')) for pm in prompt_metrics_list], float('inf'))
    aggregated_metrics['avg_orig_detect'] = safe_mean([pm.get('original_detection', DETECTION_SCORE_FLOOR_IN_OBJECTIVE) for pm in prompt_metrics_list], DETECTION_SCORE_FLOOR_IN_OBJECTIVE)
    aggregated_metrics['avg_attack_detect'] = safe_mean([pm.get('attacked_detection', DETECTION_SCORE_FLOOR_IN_OBJECTIVE) for pm in prompt_metrics_list], DETECTION_SCORE_FLOOR_IN_OBJECTIVE)
    aggregated_metrics['avg_unwatermarked_detect'] = safe_mean([pm.get('unwatermarked_detection', DETECTION_SCORE_FLOOR_IN_OBJECTIVE) for pm in prompt_metrics_list], DETECTION_SCORE_FLOOR_IN_OBJECTIVE)

    robustness_diff_val = aggregated_metrics['avg_attack_detect'] - aggregated_metrics['avg_unwatermarked_detect']
    if np.isnan(robustness_diff_val) or np.isinf(robustness_diff_val): # 检查inf
        robustness_diff_val = DETECTION_SCORE_FLOOR_IN_OBJECTIVE - 1.0 # 或其他合适的差值
    aggregated_metrics['robustness_diff'] = robustness_diff_val

    # 计算一个“目标分数”用于记录，但网格搜索不依赖它进行优化
    calculated_objective_score = calculate_objective_score_paper_centric(
        metrics_dict=aggregated_metrics,
        weights=OBJECTIVE_WEIGHTS_PAPER_CENTRIC,
        penalty_for_invalid_metric=PENALTY_FOR_INVALID_METRIC_IN_OBJECTIVE,
        detection_score_floor=DETECTION_SCORE_FLOOR_IN_OBJECTIVE
    )
    aggregated_metrics['calculated_objective_score'] = calculated_objective_score

    trial_time_taken = time.time() - trial_start_time
    print(f"    Finished Evaluation: alpha={current_alpha}, delta={current_delta:.1f}, Time={trial_time_taken:.2f}s")
    print(f"      Avg Metrics: PPL_wm={aggregated_metrics['avg_ppl_watermarked']:.2f}, PPL_base={aggregated_metrics['avg_ppl_unwatermarked_baseline']:.2f}, PPL_rel={aggregated_metrics['avg_ppl_relative']:.3f}, OrigDet={aggregated_metrics['avg_orig_detect']:.3f}, AttackDetect={aggregated_metrics['avg_attack_detect']:.3f}, UnwmDetect={aggregated_metrics['avg_unwatermarked_detect']:.3f}, RobDiff={aggregated_metrics['robustness_diff']:.3f}, ObjScore={calculated_objective_score:.4f}")
    return aggregated_metrics # 只返回指标字典

# --- Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()
    print("=== Starting Hyperparameter Grid Search (Data Collection Mode) ===")

    LOADED_MODELS_CACHE = load_all_models_once()
    if LOADED_MODELS_CACHE is None:
        print("FATAL ERROR: Model loading failed. Exiting script.")
        exit()

    print("\n--- Pre-calculating Baseline PPLs for test prompts ---")
    if LOADED_MODELS_CACHE:
        temp_watermark_obj_for_baseline = Watermark(
                device=LOADED_MODELS_CACHE['device'], watermark_tokenizer=LOADED_MODELS_CACHE['watermark_tokenizer'],
                measure_tokenizer=LOADED_MODELS_CACHE['measure_tokenizer'], watermark_model=LOADED_MODELS_CACHE['watermark_model'],
                measure_model=LOADED_MODELS_CACHE['measure_model'], embedding_model=LOADED_MODELS_CACHE['embedding_model'],
                transform_model=LOADED_MODELS_CACHE['transform_model'], mapping_list=LOADED_MODELS_CACHE['mapping_list'],
                alpha=ALPHA_GRID[0], delta=DELTA_GRID[0], **GENERATION_CONFIG )
        try:
            with open(TEST_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                ALL_TEST_PROMPTS_CACHE = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if not ALL_TEST_PROMPTS_CACHE or len(ALL_TEST_PROMPTS_CACHE) < NUM_TEST_PROMPTS_TO_USE:
                print(f"ERROR: Not enough usable prompts. Need {NUM_TEST_PROMPTS_TO_USE}, Found {len(ALL_TEST_PROMPTS_CACHE)}.")
                exit()
            prompts_for_baseline = ALL_TEST_PROMPTS_CACHE[:NUM_TEST_PROMPTS_TO_USE]
            for idx, prompt_text in enumerate(prompts_for_baseline):
                print(f"  Pre-calculating baseline for prompt {idx+1}/{len(prompts_for_baseline)}...")
                get_or_calculate_baseline_ppl(prompt_text, temp_watermark_obj_for_baseline, LOADED_MODELS_CACHE)
            print("--- Baseline PPLs pre-calculation finished. ---")
        except FileNotFoundError: print(f"ERROR: Test prompts file not found: '{TEST_PROMPTS_FILE}'"); exit()
        except Exception as e: print(f"ERROR loading prompts or pre-calculating baseline PPLs: {e}"); exit()
    else: print("ERROR: Models not loaded, cannot pre-calculate baseline PPLs."); exit()

    print("\n--- Starting Grid Search Data Collection ---")
    current_point_count = 0
    for alpha_param in ALPHA_GRID:
        for delta_param in DELTA_GRID:
            current_point_count += 1
            print(f"\n--- Evaluating Grid Point {current_point_count}/{TOTAL_GRID_POINTS}: alpha={alpha_param}, delta={delta_param:.1f} ---")

            # 直接调用评估函数获取所有指标
            metrics = evaluate_single_params_for_grid( # 使用新的评估函数名
                alpha_val=alpha_param,
                delta_val=delta_param,
                current_loaded_models=LOADED_MODELS_CACHE,
                current_test_prompts=ALL_TEST_PROMPTS_CACHE
            )

            trial_data = {
                'alpha': alpha_param,
                'delta': delta_param,
            }
            # 将 metrics 字典中的所有内容添加到 trial_data
            trial_data.update(metrics)
            ALL_GRID_SEARCH_RESULTS.append(trial_data)

    print("\n--- Grid Search Data Collection Finished ---")

    if ALL_GRID_SEARCH_RESULTS:
        results_df = pd.DataFrame(ALL_GRID_SEARCH_RESULTS)
        cols_to_display = [
            'alpha', 'delta',
            'avg_ppl_watermarked',
            'avg_ppl_unwatermarked_baseline',
            'avg_ppl_relative',
            'avg_orig_detect',
            'avg_attack_detect',
            'avg_unwatermarked_detect',
            'robustness_diff',
            'calculated_objective_score' # 之前目标函数计算的分数
        ]
        for col in cols_to_display: # 确保所有列都存在
            if col not in results_df.columns:
                results_df[col] = np.nan
        results_df_display = results_df[cols_to_display]

        # 为人工选择排序，例如，优先考虑鲁棒性，然后PPL，然后原始检测
        results_df_sorted = results_df_display.sort_values(
            by=['robustness_diff', 'avg_ppl_relative', 'avg_orig_detect'], # 示例排序
            ascending=[False, True, False] # rob_diff越大越好, ppl_rel越小越好, orig_det越大越好
        )

        print("\n--- Grid Search Results Table (Sorted for Manual Selection) ---")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200) # 适当增加宽度以便更好地显示多列
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(results_df_sorted.to_string(index=False))

        results_filename = f"grid_search_DATA_COLLECTION_{WATERMARK_MODEL_PATH.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        try:
            results_df_sorted.to_csv(results_filename, index=False, float_format='%.4f')
            print(f"\nAll grid search results saved to: {results_filename}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
    else:
        print("No results from grid search data collection.")

    main_end_time = time.time()
    print(f"\nTotal script execution time: {(main_end_time - main_start_time)/60:.2f} minutes.")
    print("=== Script Finished ===")