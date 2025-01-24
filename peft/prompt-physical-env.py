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


# prompt = """
# 你将获得由用户提供的文本，检查是否符合等级保护测评中，机房场地选择在具有防震、防风和防雨能力的建筑内指标的操作步骤，对于每个步骤要点，请进行如下判断：
# 1 -陈述指标的正确步骤，判断是否满足该步骤要求。
# 2 -满足则输出"yes"，不满足请说明原因。

# 最后，提供有多少个"yes"回答。将此计数作为{"count":<insert count here>}。

# 用户：1.建筑物抗震设有防审批文档。
# 2.没有雨水渗漏的痕迹。
# 3.有方便开启的窗户。
# 4.屋顶、墙体、门窗和地面没有破损开裂的情况。
# """


# prompt = """
# 在进行等级保护测评中，对于物理位置选择的安全控制点下的机房场地应选择在具有防震、防风和防雨能力的建筑内，具体需要执行哪些操作步骤来确保符合这一要求？

# 对于每个步骤要点，与用户输入的几点进行对比进行如下判断：
# 1 -陈述指标的正确步骤，判断是否满足该步骤要求。
# 2 -满足则输出"yes"，不满足请说明原因。

# 最后，提供有多少个"yes"回答。将此计数作为{"count":<insert count here>}。

# 用户：1.建筑物抗震设有防审批文档。
# 2.没有雨水渗漏的痕迹。
# 3.有方便开启的窗户。
# 4.屋顶、墙体、门窗和地面没有破损开裂的情况。
# """

prompt = """
在进行等级保护测评中，对于物理位置选择的安全控制点下的机房场地应选择在具有防震、防风和防雨能力的建筑内，具体需要执行哪些操作步骤来确保符合这一要求？
"""

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

ft_response = compare_chatglm_results(prompt, qlora_model, training_tag)



# # 使用模型生成文本
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)

# # 解码生成的文本
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 输出结果
# print(generated_text)