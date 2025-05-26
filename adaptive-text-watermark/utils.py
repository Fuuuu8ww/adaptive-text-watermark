# utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random


def load_model(model_name_or_path, device=None): # 修改点1：增加 device 参数，并将 model_name 改为 model_name_or_path
    """
    Loads a Hugging Face tokenizer and model from a given name or local path.

    Args:
        model_name_or_path (str): The name of the model (e.g., 'gpt2-large') or
                                  the local path to the model directory.
        device (torch.device, optional): The device to load the model onto.
                                         If None, defaults to 'cuda' if available, else 'cpu'.

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if device is None: # 修改点2：如果未指定device，则自动选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {model_name_or_path} to device: {device}")

    # 修改点3：移除 device_map='auto'，使用 .to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    model.eval()
    print(f"Model successfully loaded to: {model.device}") # 确认模型设备
    return model, tokenizer

def vocabulary_mapping(vocab_size, model_output_dim, seed=66):
    random.seed(seed)
    return [random.randint(0, model_output_dim-1) for _ in range(vocab_size)]

# pre_process 函数通常用于数据处理，与模型加载关系不大，暂时可以不修改，除非你发现它也需要设备相关的处理
def pre_process(dataset, min_length, data_size=500):
    data = []
    for text in dataset['train']['text']:
        text0 = text.split()[0:min_length]
        if len(text0) >= min_length:
            text0 = ' '.join(text0)
            data.append({'text0': text0, 'text': text})
        else:
            pass

        if len(data) ==  data_size:
            break

    return data
