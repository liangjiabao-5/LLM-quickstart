import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os


# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 设置本地缓存目录
os.environ['HF_HUB_CACHE'] = '/mnt/new_volume/hf/hub'

# 指定本地模型路径
local_model_path = os.path.join(os.environ['HF_HUB_CACHE'], 'chatglm3-6b')



q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)

base_model = AutoModel.from_pretrained(local_model_path,
                                  quantization_config=q_config,
                                  device_map='auto',
                                  trust_remote_code=True)

base_model.requires_grad_(False)
base_model.eval()


input_text = '请问在等级保护测评过程中，如何评估机房场地选择在具有防震、防风和防雨能力的建筑内是否满足安全物理环境的要求？'
print(f'输入：\n{input_text}')

# revision='b098244' 版本对应的 ChatGLM3-6B 设置 use_reentrant=False
# 最新版本 use_reentrant 被设置为 True，会增加不必要的显存开销
tokenizer = AutoTokenizer.from_pretrained(local_model_path,
                                          trust_remote_code=True,
                                          revision='b098244')

# response, history = base_model.chat(tokenizer=tokenizer, query=input_text)
# print(f'ChatGLM3-6B 微调前：\n{response}')

print("------------------------------------------------------------------------------------------------")
# # 定义全局变量和参数
epochs = 3

timestamp = "20240821_105152"
model_name_or_path = 'THUDM/chatglm3-6b'  # 模型ID或本地路径
book = 'Secure_physical_environment'
peft_model_path = f"models/{book}/{model_name_or_path}-epoch{epochs}-{timestamp}"


config = PeftConfig.from_pretrained(peft_model_path)
qlora_model = PeftModel.from_pretrained(base_model, peft_model_path)
training_tag=f"ChatGLM3-6B(Epoch=3, automade-dataset(fixed))-{timestamp}"


def compare_chatglm_results(query, qlora_model, training_tag):

    inputs = tokenizer(query, return_tensors="pt").to(0)
    ft_out = qlora_model.generate(**inputs, max_new_tokens=512)
    ft_response = tokenizer.decode(ft_out[0], skip_special_tokens=True)
    
    print(f"ChatGLM3-6B 微调后：\n{ft_response}")
    return ft_response

ft_response = compare_chatglm_results(input_text, qlora_model, training_tag)


# inference_model = AutoModel.from_pretrained(config.base_model_name_or_path,
#                                        quantization_config=q_config,
#                                        trust_remote_code=True,
#                                        device_map='auto')
# inference_model.requires_grad_(False)
# inference_model.eval()

# inference_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

# response, history = inference_model.chat(tokenizer=inference_tokenizer, query=input_text)
# print(f'ChatGLM3-6B 微调后: \n{response}')