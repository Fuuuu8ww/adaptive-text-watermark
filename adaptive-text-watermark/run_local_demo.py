# run_local_demo.py

import torch
from sentence_transformers import SentenceTransformer
# 假设 model.py, utils.py, watermark.py 都在当前目录下或Python路径可达
from model import SemanticModel
from utils import load_model, vocabulary_mapping
from watermark import Watermark # 注意：这是你提供的Watermark类定义所在的文件名

# --- BEGIN: NLTK Path Debugging ---
import nltk
import os
from nltk.data import path as nltk_data_path_list # 直接访问路径列表
from nltk.data import find as nltk_find # 导入 NLTK 的 find 函数

print("--- NLTK Debugging ---")
nltk_data_env_var = os.getenv('NLTK_DATA')
print(f"Value of NLTK_DATA environment variable: {nltk_data_env_var}")

custom_path = "/home/szr/nltk_data"
if os.path.exists(custom_path):
    if custom_path in nltk_data_path_list:
        nltk_data_path_list.remove(custom_path)
    nltk_data_path_list.insert(0, custom_path)
    print(f"Forcefully set '{custom_path}' as the first NLTK data path.")
else:
    print(f"Warning: Custom NLTK data path does not exist: {custom_path}")

print(f"NLTK's data.path being used for find(): {nltk.data.path}") # nltk.data.find 会用这个列表

resource_name_to_find = "tokenizers/punkt_tab/english/"
print(f"\nAttempting to find resource '{resource_name_to_find}' using nltk.data.find()...")
try:
    # nltk.data.find() 会在 nltk.data.path 中搜索
    # 它期望找到一个文件或一个包含特定标记文件的目录
    # PunktTokenizer 会尝试加载 'english.pickle'
    # 所以我们尝试直接找 'tokenizers/punkt_tab/english.pickle' (虽然它实际是目录)
    # 或者更准确地说，PunktTokenizer 会用 find('tokenizers/punkt_tab/{lang}/')
    # 然后期望这个路径是一个目录，里面有 'english.pickle'
    found_resource_path = nltk_find(resource_name_to_find) # find 会在成功时返回ZipFilePathPointer或FileSystemPathPointer
    print(f"  nltk.data.find() found resource at: {found_resource_path}")
    # 进一步检查这个路径是否真的是一个包含我们期望文件的目录
    # found_resource_path 通常是一个指向目录的指针对象
    # 我们可以尝试列出这个目录下的内容，或者检查特定文件是否存在
    # 注意：FileSystemPathPointer 对象可以直接用 os.path.join
    expected_pickle = "english.pickle"
    if hasattr(found_resource_path, 'path') and os.path.isdir(found_resource_path.path): # 对于FileSystemPathPointer
        actual_dir_path = found_resource_path.path
    elif isinstance(found_resource_path, str) and os.path.isdir(found_resource_path): # 如果直接返回字符串路径
        actual_dir_path = found_resource_path
    else: # 对于ZipFilePathPointer，处理会更复杂，但我们这里是本地文件系统
        actual_dir_path = str(found_resource_path) # 尝试转换为字符串

    if os.path.isdir(actual_dir_path) and expected_pickle in os.listdir(actual_dir_path):
        print(f"  Confirmed: Directory '{actual_dir_path}' exists and contains '{expected_pickle}'.")
    else:
        print(f"  Warning: Found path '{actual_dir_path}' but it's not a directory or doesn't contain '{expected_pickle}'.")

except LookupError as e:
    print(f"  nltk.data.find() raised a LookupError (as expected if not found): {e}")
except Exception as e:
    print(f"  nltk.data.find() raised an unexpected error: {e}")


print("--- End NLTK Debugging ---")
# --- END: NLTK Path Debugging ---

# --- 1. 定义你的本地模型路径 ---
# !!! 请根据你服务器上的实际路径仔细检查并修改 !!!
BASE_MODEL_PATH = "/raid_sdh/home/szr/" # 你的模型根目录

OPT_MODEL_PATH = f"{BASE_MODEL_PATH}opt-6.7b"
GPT2_MODEL_PATH = f"{BASE_MODEL_PATH}gpt-large-2" # 注意，README中是 gpt2-large
SBERT_MODEL_PATH = f"{BASE_MODEL_PATH}sentence-transformers_all-mpnet-base-v2" # 你保存的SBERT文件夹
# SemanticMappingModel的路径，它在代码仓库的model/文件夹下
# 假设 run_local_demo.py 在仓库根目录
SEMANTIC_MAPPING_MODEL_PATH = "model/semantic_mapping_model.pth"

# --- 2. 设置设备 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 3. 加载模型 (使用本地路径) ---
print(f"Loading watermarking model from: {OPT_MODEL_PATH}")
watermark_model, watermark_tokenizer = load_model(OPT_MODEL_PATH, device=device) # 传入device

print(f"Loading measurement model from: {GPT2_MODEL_PATH}")
measure_model, measure_tokenizer = load_model(GPT2_MODEL_PATH, device=device) # 传入device

print(f"Loading semantic embedding model from: {SBERT_MODEL_PATH}")
embedding_model = SentenceTransformer(SBERT_MODEL_PATH, device=device) # SentenceTransformer可以直接用路径和device
embedding_model.eval()

print(f"Loading semantic mapping model from: {SEMANTIC_MAPPING_MODEL_PATH}")
transform_model = SemanticModel()
# 使用 map_location=device 确保模型正确加载到指定设备
transform_model.load_state_dict(torch.load(SEMANTIC_MAPPING_MODEL_PATH, map_location=device))
transform_model.to(device) # 再次确保模型在正确设备上
transform_model.eval()

print("Loading mapping list...")
# --- 修改点在这里 ---
# 使用 watermark_model.config.vocab_size 而不是 watermark_tokenizer.vocab_size
actual_vocab_size = watermark_model.config.vocab_size
print(f"Using model's actual vocab size for mapping: {actual_vocab_size}") # 确认这个值是不是 50272

mapping_dim = 384
mapping_list = vocabulary_mapping(actual_vocab_size, mapping_dim, seed=66)
print(f"Actual vocabulary size used for mapping_list: {len(mapping_list)}, Mapping dimension: {mapping_dim}")
# --- 修改结束 ---


# --- 4. 初始化 Watermark 对象 (使用README中的参数) ---
print("Initializing Watermark object...")
# 注意：提供的 Watermark 类定义中，构造函数参数名与README略有不同
# 比如README用 alpha=2, 而类定义用 self.alpha 对应熵阈值 alpha
# README用 delta=1.5, 类定义用 self.delta 对应水印强度 delta
# README用 measure_threshold=50, 类定义用 self.measure_threshold 对应序列长度阈值
# 我们将按照Watermark类定义中的参数名来初始化，但使用README中给出的值，并明确对应关系

watermark_obj = Watermark( # 类名是Watermark
    device=device,
    watermark_tokenizer=watermark_tokenizer,
    measure_tokenizer=measure_tokenizer,
    watermark_model=watermark_model,
    measure_model=measure_model,
    embedding_model=embedding_model,
    transform_model=transform_model,
    mapping_list=mapping_list,
    alpha=2.0,  # 这是论文中的熵阈值 alpha (类中也叫alpha)
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0, # 根据类定义，默认为0
    max_new_tokens=230,
    min_new_tokens=170,
    secret_string='The quick brown fox jumps over the lazy dog', # 类定义中有此参数
    measure_threshold=50, # 这是论文中的序列长度阈值 (类中也叫measure_threshold)
    delta_0=1.0, # 类定义中有此参数
    delta=1.5,   # 这是论文中的水印强度 delta (类中也叫delta)
)
print("Watermark object initialized.")

# --- 5. 生成和检测文本 ---
prompt = "The field of artificial intelligence has seen rapid advancements in recent years" # 使用一个稍微长一点的prompt
print(f"\nUsing prompt: '{prompt}'")

print("\nGenerating un-watermarked text...")
unwatermarked_text = watermark_obj.generate_unwatermarked(prompt) # 使用watermark_obj
print(f"Unwatermarked text: {unwatermarked_text}")

print("\nGenerating watermarked text...")
# 注意README中是 generate_adaptive_watermarke，而你提供的类定义中有 generate_adaptive_watermarke 和 generate_watermarked
# generate_adaptive_watermarke 内部调用了 generate_watermarked 并增加了重采样逻辑
watermarked_text = watermark_obj.generate_adaptive_watermarke(prompt) # 使用watermark_obj
print(f"Watermarked text: {watermarked_text}")

print("\nDetecting un-watermarked text...")
unwatermark_score = watermark_obj.detection(unwatermarked_text) # 使用watermark_obj
print(f"Unwatermarked text detection score: {unwatermark_score}")

print("\nDetecting watermarked text...")
watermark_score = watermark_obj.detection(watermarked_text) # 使用watermark_obj
print(f"Watermarked text detection score: {watermark_score}")

print("\n--- Understanding the Detection Score ---")
print("The detection score from `watermark_obj.detection(text)` is a normalized value.")
print("It is calculated by:")
print("1. For each token in the input text:")
print("   a. If the token index is <= `self.measure_threshold` (e.g., 50):")
print("      - An embedding `ve` is derived from `self.secret_string` via `self.embedding_model` and `self.transform_model`.")
print("      - The score for this token is `ve[token_id]` (which will be 0.0 or 1.0).")
print("   b. If the token index is > `self.measure_threshold`:")
print("      - The text up to the previous token (`measure_text`) is used.")
print("      - The entropy of the next token prediction for `measure_text` is calculated using `self.measure_model` (`measure_entropy`).")
print("      - If `measure_entropy >= self.alpha` (e.g., 2.0):")
print("         - An embedding `ve` is derived from `measure_text`.")
print("         - The score for this token is `ve[token_id]` (0.0 or 1.0).")
print("      - Otherwise (low entropy), this token does not contribute to the `score` list directly (it's skipped for scoring).")
print("2. All collected token scores (0s and 1s) are summed up.")
print("3. This sum is then divided by the *number of tokens that contributed to the score* (i.e., `len(score)` in the code).")
print("Therefore, the score represents the proportion of 'selected for scoring' tokens that aligned with the dynamically generated 'green list' (where `ve[token_id]` was 1.0).")
print("A higher score indicates a stronger presence of the watermark signal according to this specific detection logic.")
print("Ideally, for unwatermarked text, this score should be close to the expected random chance (e.g., if the green list typically covers 50% of the vocab, a random text might score near 0.5 on a long enough sample of scored tokens, though it can vary).")
print("For watermarked text, the score should be significantly higher, approaching 1.0 if the watermark is strong and effective.")