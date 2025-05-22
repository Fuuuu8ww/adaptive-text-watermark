# optimizer_basic.py

import os
import time
import torch
import numpy as np
import pandas as pd # For saving results
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from sentence_transformers import SentenceTransformer as SentenceTransformerLib # <--- 修改这里
# Import necessary components from your other project files
# Ensure these files (watermark.py, evaluation_utils.py, model.py, utils.py) 
# are in the same directory or accessible via PYTHONPATH.
from watermark import Watermark
from evaluation_utils import (
    calculate_ppl, 
    apply_local_paraphrase_attack, 
    calculate_objective_score_weighted_sum
) 
from model import SemanticModel # Assuming SemanticModel is defined in model.py
try:
    from utils import vocabulary_mapping # Assuming vocabulary_mapping is defined in utils.py
except ImportError:
    print("ERROR: Could not import 'vocabulary_mapping' from utils.py. Ensure it's available.")
    exit()


# --- 1. Configuration Section ---
print("--- Optimizer Configuration ---")

# Model Paths (IMPORTANT: Replace placeholders with actual paths/names on your server)
BASE_MODEL_PATH = "/data2/szr/" 
WATERMARK_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "opt-305m") # Example: your OPT-6.7B folder
MEASURE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "gpt-large-2") # Example: your GPT2-Large folder
EMBEDDING_MODEL_NAME = os.path.join(BASE_MODEL_PATH, "sentence-transformers_all-mpnet-base-v2") # Can be HF name if you downloaded it with this name structure, or a direct local path
SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH = os.path.join(BASE_MODEL_PATH, "adaptive-text-watermark", "model", "semantic_mapping_model.pth") # Example: path to the .pth file from the original repo if you use that structure
LOCAL_ATTACK_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "adaptive-text-watermark", "T5-paragraph") # Folder where Vamsi/T5_Paraphrase_Paws is saved locally
PPL_EVAL_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "Meta-Llama-3-8B-Instruct-hf")

print(f"BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print(f"WATERMARK_MODEL_PATH: {WATERMARK_MODEL_PATH}")
print(f"MEASURE_MODEL_PATH: {MEASURE_MODEL_PATH}")
print(f"EMBEDDING_MODEL_NAME (or path): {EMBEDDING_MODEL_NAME}")
print(f"SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH: {SEMANTIC_MAPPING_MODEL_WEIGHTS_PATH}")
print(f"LOCAL_ATTACK_MODEL_PATH: {LOCAL_ATTACK_MODEL_PATH}")


# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Search Space for Bayesian Optimization (scikit-optimize)
from skopt.space import Integer, Real
SPACE = [
    Integer(1, 4, name='alpha'),        # Alpha: Integer from 1 to 4
    Real(0.5, 2.0, name='delta')        # Delta: Continuous from 0.5 to 2.0 (will be rounded to 1 decimal place)
]
print(f"Search Space: Alpha from {SPACE[0].low} to {SPACE[0].high}, Delta from {SPACE[1].low} to {SPACE[1].high}")

# Bayesian Optimization Settings
N_OPTIMIZATION_CALLS = 50  # Total number of times to evaluate the objective function
N_INITIAL_RANDOM_POINTS = 10 # Number of random points to explore before starting Bayesian optimization
print(f"Bayesian Optimization: N_OPTIMIZATION_CALLS={N_OPTIMIZATION_CALLS}, N_INITIAL_RANDOM_POINTS={N_INITIAL_RANDOM_POINTS}")

# Test Prompts File
TEST_PROMPTS_FILE = "test_prompts.txt" 
NUM_TEST_PROMPTS_TO_USE = 15
print(f"Test Prompts: Using '{TEST_PROMPTS_FILE}', evaluating first {NUM_TEST_PROMPTS_TO_USE} prompts.")

# <<< 改进点 PPL_AVG_1: 为PPL平均化添加采样次数配置 >>>
N_PPL_SAMPLES = 3 # PPL平均化的采样次数 (基线和水印文本都用这个)
print(f"PPL Averaging: Using {N_PPL_SAMPLES} samples for PPL calculation.")

# Watermark Generation Parameters (Fixed during search)
GENERATION_CONFIG = {
    'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.1, 'no_repeat_ngram_size': 0,
    'max_new_tokens': 200, 'min_new_tokens': 170,
    'secret_string': 'The quick brown fox jumps over the lazy dog',
    'measure_threshold': 50, 'delta_0': 1.0
}

# PPL Calculation Configuration
PPL_MAX_MODEL_LENGTH = 1024 # Default, will try to update from watermark_model.config

# Attack Model Configuration (for Vamsi/T5_Paraphrase_Paws)
ATTACK_MAX_INPUT_LENGTH = 256
ATTACK_MAX_OUTPUT_LENGTH = 256
ATTACK_DO_SAMPLE = True
ATTACK_TOP_K = 120
ATTACK_TOP_P = 0.95
ATTACK_EARLY_STOPPING = True
ATTACK_NUM_BEAMS = 3

# <<< 改进点 OPT_1: 调整权重名称以匹配 evaluation_utils.py 中的新成本项 >>>
OBJECTIVE_WEIGHTS = {
    'w_ppl_relative': 1.0,          # 权重对应相对PPL成本
    'w_original_detection': 1.0,    # 权重对应原始检测成本 (1 - score) 或 (-score)
    'w_robustness_diff': 1.2,       # 权重对应鲁棒性差异成本 (1 - diff) 或 (-diff)
    # 可以根据需要添加其他指标的权重，例如 'w_robustness_drop': 1.0
}
PENALTY_INVALID_PPL = 10000.0 # 这个惩罚值可能需要根据归一化后的尺度调整，或者在归一化前施加
DETECTION_SCORE_FLOOR = 0.0
print(f"Objective Weights: PPL_relative={OBJECTIVE_WEIGHTS['w_ppl_relative']}, OrigDetect={OBJECTIVE_WEIGHTS['w_original_detection']}, RobustnessDiff={OBJECTIVE_WEIGHTS['w_robustness_diff']}")

# --- Global variables for models and prompts (for objective function access) ---
# Note: Using globals isn't always best practice, but simplifies for this script.
# Alternatives: functools.partial or making the optimizer part of a class.
LOADED_MODELS_CACHE = None
ALL_TEST_PROMPTS_CACHE = None

# <<< 改进点 OPT_2: 新增全局变量用于缓存无水印PPL和归一化统计数据 >>>
BASELINE_PPL_CACHE = {} # prompt_text -> ppl_score
NORMALIZATION_STATS = { # 用于存储min/max值进行归一化
    'ppl_relative': {'min': float('inf'), 'max': float('-inf')},
    'cost_orig_detect': {'min': float('inf'), 'max': float('-inf')}, # orig_detect_score 0-1 -> cost -1 to 0 or 0 to 1
    'cost_robust_diff': {'min': float('inf'), 'max': float('-inf')}  # robust_diff_score -1 to 1 -> cost -1 to 1 or 0 to 2
    # 可以为其他成本项添加
}
NORMALIZATION_WINDOW_SIZE = 20 # 使用最近 N 次迭代的数据来更新min/max (可选)
RECENT_COSTS_FOR_NORM = {key: [] for key in NORMALIZATION_STATS.keys()} # (可选)

# --- 2. Model Loading Function ---
def load_all_models_once():
    """Loads all required models and tokenizers once and caches them."""
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
        print(f"  Current PPL_MAX_MODEL_LENGTH (before loading PPL eval model): {PPL_MAX_MODEL_LENGTH}")
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
        # models_dict['embedding_model'].eval() # SentenceTransformer may not have .eval() or it's not needed
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
    
    # <<< 改进点3 开始 >>>
    try:
        print(f"Loading PPL Evaluation Model & Tokenizer from: {PPL_EVAL_MODEL_PATH} ...")
        models_dict['ppl_eval_tokenizer'] = AutoTokenizer.from_pretrained(PPL_EVAL_MODEL_PATH)
        models_dict['ppl_eval_model'] = AutoModelForCausalLM.from_pretrained(
            PPL_EVAL_MODEL_PATH,
            torch_dtype=torch.bfloat16, # 或者 torch.float16，根据你的GPU和模型支持
            device_map="auto" # 自动分配到可用GPU，如果模型较大
            # trust_remote_code=True # 如果模型需要
        ).eval() # .to(DEVICE) 被 device_map="auto" 替代，如果模型加载到多GPU或CPU+GPU
                  # 如果只用单GPU且显存足够，可以用 .to(DEVICE)
        
        # 更新 PPL_MAX_MODEL_LENGTH，如果 PPL 评估模型有不同的上下文长度
        try:
            PPL_MAX_MODEL_LENGTH_NEW = models_dict['ppl_eval_model'].config.max_position_embeddings
            # 考虑 LLaMA 3 的默认 8k 上下文，但 tokenizer 可能有 model_max_length 限制
            if hasattr(models_dict['ppl_eval_tokenizer'], 'model_max_length'):
                 PPL_MAX_MODEL_LENGTH_NEW = min(PPL_MAX_MODEL_LENGTH_NEW, models_dict['ppl_eval_tokenizer'].model_max_length)

            if PPL_MAX_MODEL_LENGTH_NEW != PPL_MAX_MODEL_LENGTH:
                 PPL_MAX_MODEL_LENGTH = PPL_MAX_MODEL_LENGTH_NEW
                 print(f"  PPL max length updated from PPL evaluation model config: {PPL_MAX_MODEL_LENGTH}")
            else:
                 print(f"  PPL max length remains: {PPL_MAX_MODEL_LENGTH} (either from watermark model or default)")

        except AttributeError:
            print(f"  Warning: Could not get max_position_embeddings from PPL evaluation model. Using PPL max length: {PPL_MAX_MODEL_LENGTH}")
        print("  PPL Evaluation Model & Tokenizer: OK")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load PPL Evaluation Model/Tokenizer from '{PPL_EVAL_MODEL_PATH}': {e}")
        # 可以选择返回 None，或者让PPL评估优雅降级（例如使用watermark_model评估PPL，但不推荐）
        return None # 或者采取其他错误处理

    # 6. Generate mapping_list BASED ON watermark_model.config.vocab_size
    try:
        # ================================================================================ #
        # ========================== CRITICAL MODIFICATION HERE ========================== #
        # ================================================================================ #
        # We MUST use the vocab_size from the watermark_model's config, as this defines
        # the dimension of the logits that the mapping_list (via v_embedding) will interact with.
        
        if 'watermark_model' not in models_dict or not hasattr(models_dict['watermark_model'], 'config') or not hasattr(models_dict['watermark_model'].config, 'vocab_size'):
            print("FATAL ERROR: watermark_model or its config or vocab_size is not available for generating mapping_list.")
            return None
            
        vocab_size_for_mapping = models_dict['watermark_model'].config.vocab_size # This should be 50272 for OPT-6.7B
        
        print(f"  Using watermark_model.config.vocab_size ({vocab_size_for_mapping}) for generating mapping_list.")
        
        models_dict['mapping_list'] = vocabulary_mapping(vocab_size_for_mapping, 384, seed=66)
        print(f"  Mapping list generated. Length: {len(models_dict['mapping_list'])} (should match {vocab_size_for_mapping}).")
        # ================================================================================ #
        # ================================================================================ #
    except Exception as e:
        print(f"FATAL ERROR: Failed to generate mapping list: {e}")
        return None

    models_dict['device'] = DEVICE
    # 添加这些调试打印！！！
    print(f"DEBUG (load_all_models_once end): models_dict['watermark_tokenizer'].vocab_size = {models_dict['watermark_tokenizer'].vocab_size if 'watermark_tokenizer' in models_dict and hasattr(models_dict['watermark_tokenizer'], 'vocab_size') else 'N/A'}")
    print(f"DEBUG (load_all_models_once end): models_dict['measure_tokenizer'].vocab_size = {models_dict['measure_tokenizer'].vocab_size if 'measure_tokenizer' in models_dict and hasattr(models_dict['measure_tokenizer'], 'vocab_size') else 'N/A'}")
    if 'mapping_list' in models_dict and models_dict['mapping_list'] is not None:
        print(f"DEBUG (load_all_models_once end): len(models_dict['mapping_list']) = {len(models_dict['mapping_list'])}")
    else:
        print("DEBUG (load_all_models_once end): models_dict['mapping_list'] is None or not found.")

    print("--- All available models loaded successfully. ---")
    LOADED_MODELS_CACHE = models_dict
    return models_dict

# <<< 改进点 OPT_3_EXT: 函数获取或计算基线PPL，并返回代表性文本 >>>
def get_or_calculate_baseline_ppl(prompt_text, watermark_obj_for_unwatermarked_gen, loaded_models):
    global BASELINE_PPL_CACHE, N_PPL_SAMPLES
    # 缓存的键可以是 prompt_text，值是一个元组 (avg_ppl, representative_text)
    cache_key = prompt_text
    if cache_key in BASELINE_PPL_CACHE:
        return BASELINE_PPL_CACHE[cache_key] # 返回 (avg_ppl, representative_text)

    print(f"    Calculating baseline PPL (avg over {N_PPL_SAMPLES} samples) for prompt: \"{prompt_text[:50]}...\"")
    
    generated_baseline_text_samples = [] # 存储生成的文本
    prompt_baseline_ppls = []
    for _ in range(N_PPL_SAMPLES):
        unwatermarked_text_sample = watermark_obj_for_unwatermarked_gen.generate_unwatermarked(prompt_text)
        generated_baseline_text_samples.append(unwatermarked_text_sample) # 保存生成的文本

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
    
    # 选择一个代表性基线文本 (例如第一个非空样本)
    representative_baseline_text = next((txt for txt in generated_baseline_text_samples if txt and txt.strip()), "")
        
    BASELINE_PPL_CACHE[cache_key] = (avg_baseline_ppl_for_prompt, representative_baseline_text)
    print(f"      Average Baseline PPL: {avg_baseline_ppl_for_prompt:.2f}, Rep. Text: \"{representative_baseline_text[:30]}...\" (from {len(valid_ppls)} valid samples)")
    return avg_baseline_ppl_for_prompt, representative_baseline_text

# --- 3. Evaluation Function (Called by the optimizer's objective function) ---
def evaluate_single_trial(alpha_val, delta_val, current_loaded_models, current_test_prompts):
    """
    Evaluates a single (alpha, delta) pair.
    Returns: objective_score (float), metrics_dict (dict)
    """
    trial_start_time = time.time()
    # Ensure alpha is int and delta is float for Watermark class
    current_alpha = int(alpha_val)
    current_delta = float(delta_val) 
    
    print(f"  Trial: alpha={current_alpha}, delta={current_delta:.1f}")

    try:
        watermark_obj = Watermark(
            device=current_loaded_models['device'],
            watermark_tokenizer=current_loaded_models['watermark_tokenizer'],
            measure_tokenizer=current_loaded_models['measure_tokenizer'],
            watermark_model=current_loaded_models['watermark_model'],
            measure_model=current_loaded_models['measure_model'],
            embedding_model=current_loaded_models['embedding_model'],
            transform_model=current_loaded_models['transform_model'],
            mapping_list=current_loaded_models['mapping_list'],
            alpha=current_alpha, 
            delta=current_delta,
            **GENERATION_CONFIG
        )
    except Exception as e:
        print(f"    ERROR: Failed to instantiate Watermark object: {e}")
        return float('inf'), { # 返回一个包含所有期望键的字典
            'avg_ppl_watermarked': float('inf'),
            'avg_ppl_unwatermarked_baseline': float('inf'),
            'avg_ppl_relative': float('inf'),
            'avg_orig_detect': DETECTION_SCORE_FLOOR,
            'avg_attack_detect': DETECTION_SCORE_FLOOR,
            'avg_unwatermarked_detect': DETECTION_SCORE_FLOOR,
            'robustness_diff': DETECTION_SCORE_FLOOR - 1.0 # 确保所有键都存在
        }

    prompt_metrics_list = [] # 存储每个prompt的详细指标字典

    prompts_to_run = current_test_prompts[:NUM_TEST_PROMPTS_TO_USE]

    for i, prompt_text in enumerate(prompts_to_run):
        current_prompt_metrics = {}
        try:
            ppl_unwatermarked_baseline_avg, representative_baseline_text_for_detection = get_or_calculate_baseline_ppl(prompt_text, watermark_obj, current_loaded_models) # 新的调用
            current_prompt_metrics['ppl_unwatermarked_baseline'] = ppl_unwatermarked_baseline_avg

            # --- 生成N个水印文本样本并计算平均PPL ---
            generated_watermarked_text_samples = []
            for _ in range(N_PPL_SAMPLES): # 使用全局采样次数
                generated_watermarked_text_samples.append(watermark_obj.generate_adaptive_watermarke(prompt_text))

            current_prompt_watermarked_ppls = []
            for text_sample in generated_watermarked_text_samples:
                if not text_sample or not text_sample.strip():
                    current_prompt_watermarked_ppls.append(float('inf'))
                else:
                    current_prompt_watermarked_ppls.append(calculate_ppl(
                        text_sample, 
                        current_loaded_models['ppl_eval_model'], 
                        current_loaded_models['ppl_eval_tokenizer'], 
                        current_loaded_models['device'], 
                        PPL_MAX_MODEL_LENGTH
                    ))
            
            valid_ppls_for_current_wm = [p for p in current_prompt_watermarked_ppls if p != float('inf') and not np.isnan(p)]
            avg_ppl_watermarked_for_prompt = np.mean(valid_ppls_for_current_wm) if valid_ppls_for_current_wm else float('inf')
            current_prompt_metrics['ppl_watermarked'] = avg_ppl_watermarked_for_prompt
            # --- 结束水印PPL平均化 ---

            # 选择一个代表性水印文本用于后续检测和攻击 (例如，第一个非空样本)
            watermarked_text_for_downstream = next((txt for txt in generated_watermarked_text_samples if txt and txt.strip()), "")

            if not watermarked_text_for_downstream:
                print(f"    Skipping detection/attack for prompt {i+1} due to all watermarked_text samples being empty.")
                current_prompt_metrics.update({
                    'ppl_relative': float('inf') if avg_ppl_watermarked_for_prompt == float('inf') or ppl_unwatermarked_baseline_avg == float('inf') or ppl_unwatermarked_baseline_avg == 0 else (avg_ppl_watermarked_for_prompt / ppl_unwatermarked_baseline_avg) - 1.0,
                    'original_detection': DETECTION_SCORE_FLOOR,
                    'attacked_detection': DETECTION_SCORE_FLOOR,
                    # 'unwatermarked_detection': DETECTION_SCORE_FLOOR # 将在这里统一处理
                })
                # <<< OPT_MOD_1.1: 即使水印文本为空，也使用缓存的代表性基线文本进行unwatermarked_detection >>>
                if not representative_baseline_text_for_detection:
                    current_prompt_metrics['unwatermarked_detection'] = DETECTION_SCORE_FLOOR
                else:
                    unwatermarked_score = watermark_obj.detection(representative_baseline_text_for_detection)
                    current_prompt_metrics['unwatermarked_detection'] = unwatermarked_score if unwatermarked_score is not None else DETECTION_SCORE_FLOOR
                prompt_metrics_list.append(current_prompt_metrics)
                continue
            
            # 计算相对PPL (基于平均值)
            if ppl_unwatermarked_baseline_avg > 0 and ppl_unwatermarked_baseline_avg != float('inf') and avg_ppl_watermarked_for_prompt != float('inf'):
                ppl_relative = (avg_ppl_watermarked_for_prompt / ppl_unwatermarked_baseline_avg) - 1.0
            else:
                ppl_relative = float('inf') 
            current_prompt_metrics['ppl_relative'] = ppl_relative

            orig_score = watermark_obj.detection(watermarked_text_for_downstream)
            current_prompt_metrics['original_detection'] = orig_score if orig_score is not None else DETECTION_SCORE_FLOOR

            # 为 unwatermarked_detection 生成一个代表性文本 (或从基线样本中选)
            # 简化：重新生成一次。理论上应该使用与基线PPL计算时一致的文本或其平均行为的代表
            #temp_unwatermarked_text_for_detect = watermark_obj.generate_unwatermarked(prompt_text) 
            # <<< OPT_MOD_1.2: 使用从get_or_calculate_baseline_ppl获取的代表性基线文本进行unwatermarked_detection >>>
            if not representative_baseline_text_for_detection: # 再次检查以防万一
                current_prompt_metrics['unwatermarked_detection'] = DETECTION_SCORE_FLOOR
            else:
                unwatermarked_score = watermark_obj.detection(representative_baseline_text_for_detection)
                current_prompt_metrics['unwatermarked_detection'] = unwatermarked_score if unwatermarked_score is not None else DETECTION_SCORE_FLOOR

            attacked_text = apply_local_paraphrase_attack(
                watermarked_text_for_downstream, current_loaded_models.get('attack_model'), current_loaded_models.get('attack_tokenizer'),
                current_loaded_models['device'], ATTACK_MAX_INPUT_LENGTH, ATTACK_NUM_BEAMS,
                ATTACK_MAX_OUTPUT_LENGTH, ATTACK_DO_SAMPLE, ATTACK_TOP_K, ATTACK_TOP_P, ATTACK_EARLY_STOPPING
            )
            attack_score = DETECTION_SCORE_FLOOR
            if attacked_text and attacked_text.strip() and attacked_text != watermarked_text_for_downstream: # 确保攻击有效
                attack_score_val = watermark_obj.detection(attacked_text)
                if attack_score_val is not None:
                    attack_score = attack_score_val
            current_prompt_metrics['attacked_detection'] = attack_score
            prompt_metrics_list.append(current_prompt_metrics)

        except Exception as e:
            print(f"    ERROR during prompt {i+1} evaluation: {e}")
            cached_baseline_data = BASELINE_PPL_CACHE.get(prompt_text) # 获取元组或None
            baseline_ppl_for_error_case = cached_baseline_data[0] if cached_baseline_data else float('inf') # 安全提取PPL
            prompt_metrics_list.append({
                'ppl_watermarked': float('inf'),
                'ppl_unwatermarked_baseline': baseline_ppl_for_error_case, # 使用提取的PPL
                'ppl_relative': float('inf'),
                'original_detection': DETECTION_SCORE_FLOOR,
                'attacked_detection': DETECTION_SCORE_FLOOR,
                'unwatermarked_detection': DETECTION_SCORE_FLOOR
            })

    # Aggregate metrics
    def safe_mean(values, default_val):
        valid_values = [v for v in values if v is not None and v != float('inf') and not np.isnan(v)] # 增加对nan的检查
        return np.mean(valid_values) if valid_values else default_val

    avg_ppl_watermarked = safe_mean([pm['ppl_watermarked'] for pm in prompt_metrics_list], float('inf'))
    avg_ppl_unwatermarked_baseline = safe_mean([pm['ppl_unwatermarked_baseline'] for pm in prompt_metrics_list], float('inf'))
    avg_ppl_relative = safe_mean([pm['ppl_relative'] for pm in prompt_metrics_list], float('inf'))
    avg_orig_detect = safe_mean([pm['original_detection'] for pm in prompt_metrics_list], DETECTION_SCORE_FLOOR)
    avg_attack_detect = safe_mean([pm['attacked_detection'] for pm in prompt_metrics_list], DETECTION_SCORE_FLOOR)
    avg_unwatermarked_detect = safe_mean([pm['unwatermarked_detection'] for pm in prompt_metrics_list], DETECTION_SCORE_FLOOR)
    robustness_diff_val = avg_attack_detect - avg_unwatermarked_detect
    # 如果减法结果是nan (例如其中一个是nan，或者两者都是nan)
    # 或者如果其中一个不是有效的检测分数（例如，如果它们可能取到非数字的特殊值，但目前看是float或inf）
    if np.isnan(robustness_diff_val):
        robustness_diff_val = DETECTION_SCORE_FLOOR - 1.0 # 给一个代表“差”的默认值

    aggregated_metrics = {
        'avg_ppl_watermarked': avg_ppl_watermarked,
        'avg_ppl_unwatermarked_baseline': avg_ppl_unwatermarked_baseline,
        'avg_ppl_relative': avg_ppl_relative,
        'avg_orig_detect': avg_orig_detect,
        'avg_attack_detect': avg_attack_detect,
        'avg_unwatermarked_detect': avg_unwatermarked_detect,
        'robustness_diff': robustness_diff_val # 使用计算好的值
    }
    

    objective_score = calculate_objective_score_weighted_sum(
        metrics_dict=aggregated_metrics, 
        weights=OBJECTIVE_WEIGHTS,
        normalization_stats_updater=update_normalization_stats_and_get_current, 
        penalty_for_invalid_ppl=PENALTY_INVALID_PPL,
        detection_score_floor_for_cost=DETECTION_SCORE_FLOOR
    )

    trial_time = time.time() - trial_start_time
    print(f"    Finished Trial: alpha={current_alpha}, delta={current_delta:.1f}, Score={objective_score:.4f}, Time={trial_time:.2f}s")
    print(f"      Avg Metrics: PPL_wm={avg_ppl_watermarked:.2f}, PPL_base={avg_ppl_unwatermarked_baseline:.2f}, PPL_rel={avg_ppl_relative:.3f}, OrigDetect={avg_orig_detect:.3f}, AttackDetect={avg_attack_detect:.3f}, UnwmDetect={avg_unwatermarked_detect:.3f}, RobDiff={aggregated_metrics['robustness_diff']:.3f}") # 添加RobDiff打印

    return objective_score, aggregated_metrics

def update_normalization_stats_and_get_current(raw_costs_for_current_trial):
    """
    用当前试验的原始成本更新全局NORMALIZATION_STATS，并返回当前的min/max。
    raw_costs_for_current_trial: 字典，键是成本类型（如'ppl_relative'），值是当前试验的原始成本。
    """
    global NORMALIZATION_STATS, RECENT_COSTS_FOR_NORM, NORMALIZATION_WINDOW_SIZE
    current_stats_for_norm = {}

    for cost_type, current_cost_value in raw_costs_for_current_trial.items():
        if cost_type not in NORMALIZATION_STATS:
            # 对于新的成本类型，初始化
            NORMALIZATION_STATS[cost_type] = {'min': float('inf'), 'max': float('-inf')}
            RECENT_COSTS_FOR_NORM[cost_type] = []

        if current_cost_value != float('inf') and not np.isnan(current_cost_value) and current_cost_value != float('-inf'): # 确保是有效值才更新滑动窗口
            RECENT_COSTS_FOR_NORM[cost_type].append(current_cost_value)
            if len(RECENT_COSTS_FOR_NORM[cost_type]) > NORMALIZATION_WINDOW_SIZE:
                RECENT_COSTS_FOR_NORM[cost_type].pop(0)

            # 更新全局统计也只用有效值
            NORMALIZATION_STATS[cost_type]['min'] = min(NORMALIZATION_STATS[cost_type]['min'], current_cost_value)
            NORMALIZATION_STATS[cost_type]['max'] = max(NORMALIZATION_STATS[cost_type]['max'], current_cost_value)

        # 无论当前值是否有效，都从RECENT_COSTS_FOR_NORM计算窗口内的min/max和观测数
        num_obs_in_current_window = len(RECENT_COSTS_FOR_NORM[cost_type])

        if RECENT_COSTS_FOR_NORM[cost_type]: # 如果滑动窗口内有数据
            current_stats_for_norm[cost_type] = {
                'min': min(RECENT_COSTS_FOR_NORM[cost_type]),
                'max': max(RECENT_COSTS_FOR_NORM[cost_type]),
                'num_obs_in_window': num_obs_in_current_window
            }
        else: # 滑动窗口为空（例如，所有历史值都无效，或者刚开始）
             current_stats_for_norm[cost_type] = {
                'min': float('inf'), # 或者使用全局 NORMALIZATION_STATS 的值
                'max': float('-inf'),# 或者使用全局 NORMALIZATION_STATS 的值
                'num_obs_in_window': 0
             }
             # 如果希望在窗口为空时使用全局统计（如果全局统计有值的话）
             # if NORMALIZATION_STATS[cost_type]['min'] != float('inf'): # 检查全局的是否已被更新过
             #    current_stats_for_norm[cost_type]['min'] = NORMALIZATION_STATS[cost_type]['min']
             #    current_stats_for_norm[cost_type]['max'] = NORMALIZATION_STATS[cost_type]['max']


    print(f"DEBUG (NormUpdater): Returning current_stats_for_norm: {current_stats_for_norm}")
    return current_stats_for_norm

# --- 4. Objective Function for Bayesian Optimizer ---
from skopt.utils import use_named_args

# This list will store results from all trials for later analysis/saving
ALL_TRIALS_RESULTS = []

@use_named_args(SPACE)
def objective_function_for_optimizer(alpha, delta):
    """
    Wrapper for evaluate_single_trial, used by scikit-optimize's gp_minimize.
    Handles rounding delta and ensures globals are accessed.
    """
    global LOADED_MODELS_CACHE, ALL_TEST_PROMPTS_CACHE, ALL_TRIALS_RESULTS
    
    processed_delta = round(delta, 1) # Ensure delta is one decimal place

    # Call the main evaluation logic
    score, metrics_dict = evaluate_single_trial(
        alpha_val=alpha, # alpha is already Integer from SPACE
        delta_val=processed_delta, 
        current_loaded_models=LOADED_MODELS_CACHE, 
        current_test_prompts=ALL_TEST_PROMPTS_CACHE
    )
    
    # <<< 改进点 OPT_8: 在ALL_TRIALS_RESULTS中记录更全面的指标 >>>
    trial_data = {
        'alpha': int(alpha),
        'delta_requested_by_optimizer': round(delta, 3),
        'delta_processed': processed_delta,
        'objective_score': score,
    }
    # 从metrics_dict中添加所有avg_开头的指标
    for key, value in metrics_dict.items():
        if key.startswith('avg_') or key == 'robustness_diff':
            trial_data[key] = value
    ALL_TRIALS_RESULTS.append(trial_data)
    return score

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()
    print("=== Starting Hyperparameter Optimization Script (Bayesian Optimization) ===")
    
    LOADED_MODELS_CACHE = load_all_models_once()
    if LOADED_MODELS_CACHE is None:
        print("FATAL ERROR: Model loading failed. Exiting script.")
        exit()

    # <<< 改进点 OPT_9: 预计算所有测试prompt的基线PPL >>>
    # 需要一个临时的Watermark实例来生成无水印文本，可以用默认或任意 (alpha, delta)
    # 因为我们假设 generate_unwatermarked 不受这些参数影响
    print("\n--- Pre-calculating Baseline PPLs for test prompts ---")
    if LOADED_MODELS_CACHE: # 确保模型已加载
        temp_watermark_obj_for_baseline = Watermark(
                device=LOADED_MODELS_CACHE['device'],
                watermark_tokenizer=LOADED_MODELS_CACHE['watermark_tokenizer'],
                measure_tokenizer=LOADED_MODELS_CACHE['measure_tokenizer'],
                watermark_model=LOADED_MODELS_CACHE['watermark_model'],
                measure_model=LOADED_MODELS_CACHE['measure_model'],
                embedding_model=LOADED_MODELS_CACHE['embedding_model'],
                transform_model=LOADED_MODELS_CACHE['transform_model'],
                mapping_list=LOADED_MODELS_CACHE['mapping_list'],
                alpha=SPACE[0].low, # 使用任意有效值
                delta=SPACE[1].low, # 使用任意有效值
                **GENERATION_CONFIG
            )
        try:
            with open(TEST_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                ALL_TEST_PROMPTS_CACHE = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if not ALL_TEST_PROMPTS_CACHE or len(ALL_TEST_PROMPTS_CACHE) < NUM_TEST_PROMPTS_TO_USE:
                print(f"ERROR: Not enough usable prompts. Need {NUM_TEST_PROMPTS_TO_USE}, Found {len(ALL_TEST_PROMPTS_CACHE)}.")
                exit()
            
            prompts_for_baseline = ALL_TEST_PROMPTS_CACHE[:NUM_TEST_PROMPTS_TO_USE] # 只为实际使用的prompt计算基线
            for prompt_text in prompts_for_baseline:
                get_or_calculate_baseline_ppl(prompt_text, temp_watermark_obj_for_baseline, LOADED_MODELS_CACHE)
            print("--- Baseline PPLs pre-calculation finished. ---")

        except FileNotFoundError:
            print(f"ERROR: Test prompts file not found: '{TEST_PROMPTS_FILE}'")
            exit()
        except Exception as e:
            print(f"ERROR loading prompts or pre-calculating baseline PPLs: {e}")
            exit()
    else:
        print("ERROR: Models not loaded, cannot pre-calculate baseline PPLs.")
        exit()


    print("\n--- Starting Bayesian Optimization with scikit-optimize ---")
    from skopt import gp_minimize
    
    optimization_result = gp_minimize(
        func=objective_function_for_optimizer,
        dimensions=SPACE,
        n_calls=N_OPTIMIZATION_CALLS, 
        n_initial_points=N_INITIAL_RANDOM_POINTS,
        acq_func="gp_hedge", # A good default, balances exploration and exploitation
        random_state=456,    # For reproducibility
        verbose=True         # Prints progress during optimization
    )

    print("\n--- Bayesian Optimization Finished ---")
    
    if optimization_result:
        best_alpha_raw, best_delta_raw = optimization_result.x
        best_alpha_final = int(best_alpha_raw)
        best_delta_final = round(best_delta_raw, 1)

        print("\nBest Parameters Found by Bayesian Optimization:")
        print(f"  Suggested by Optimizer: alpha={best_alpha_raw}, delta_raw={best_delta_raw:.3f}")
        print(f"  Processed Parameters: Alpha={best_alpha_final}, Delta={best_delta_final:.1f}")
        print(f"  Best Objective Score Achieved: {optimization_result.fun:.4f}")

        # Find the detailed metrics for the best run from ALL_TRIALS_RESULTS
        # This is more robust than re-evaluating, as it captures the exact run.
        best_trial_info = None
        # <<< 改进点 OPT_10: 查找最佳试验信息时，考虑浮点数精度问题，并确保所有关键指标都存在 >>>
        for trial in ALL_TRIALS_RESULTS:
            if trial.get('alpha') == best_alpha_final and \
               trial.get('delta_processed') == best_delta_final and \
               abs(trial.get('objective_score', float('inf')) - optimization_result.fun) < 1e-5 :
                best_trial_info = trial
                break
        if best_trial_info:
            print("  Corresponding Average Metrics for this Best Trial:")
            print(f"    Avg PPL Watermarked: {best_trial_info.get('avg_ppl_watermarked', 'N/A'):.2f}")
            print(f"    Avg PPL Baseline: {best_trial_info.get('avg_ppl_unwatermarked_baseline', 'N/A'):.2f}")
            print(f"    Avg PPL Relative: {best_trial_info.get('avg_ppl_relative', 'N/A'):.3f}")
            print(f"    Avg Original Detection: {best_trial_info.get('avg_orig_detect', 'N/A'):.3f}")
            print(f"    Avg Attacked Detection: {best_trial_info.get('avg_attack_detect', 'N/A'):.3f}")
            print(f"    Avg Unwatermarked Detection: {best_trial_info.get('avg_unwatermarked_detect', 'N/A'):.3f}")
            print(f"    Robustness Diff: {best_trial_info.get('robustness_diff', 'N/A'):.3f}")
        else:
             print("  Note: Could not directly match best params to a trial log for detailed metrics. The score above is the primary result.")
    else:
        print("\nOptimization did not return a conclusive result.")

    # Save all trial results to a CSV file
    if ALL_TRIALS_RESULTS:
        try:
            results_df = pd.DataFrame(ALL_TRIALS_RESULTS)
            results_df = results_df.sort_values(by='objective_score', ascending=True)
            # <<< 改进点 OPT_11: 更新结果文件名以反映更改 >>>
            results_filename = f"bayesian_opt_results_PPLREL_NORM_{WATERMARK_MODEL_PATH.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            results_df.to_csv(results_filename, index=False, float_format='%.4f')
            print(f"\nAll trial results saved to: {results_filename}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            print("Dumping results to console instead:")
            # <<< 改进点 OPT_12: 更新打印列以包含新指标 >>>
            for res_idx, res in enumerate(sorted(ALL_TRIALS_RESULTS, key=lambda x: x.get('objective_score', float('inf')))):
                if res_idx < 20 or res.get('objective_score') == optimization_result.fun : # 打印前20和最佳的
                    print(f"  alpha={res.get('alpha', 'N/A')}, delta_proc={res.get('delta_processed', float('nan')):.1f}, score={res.get('objective_score', float('nan')):.4f}, "
                          f"ppl_rel={res.get('avg_ppl_relative', float('nan')):.3f}, orig_det={res.get('avg_orig_detect', float('nan')):.3f}, "
                          f"att_det={res.get('avg_attack_detect', float('nan')):.3f}, unwm_det={res.get('avg_unwatermarked_detect', float('nan')):.3f}, "
                          f"rob_diff={res.get('robustness_diff', float('nan')):.3f}")

    main_end_time = time.time()
    print(f"\nTotal script execution time: {(main_end_time - main_start_time)/60:.2f} minutes.")
    print("=== Script Finished ===")