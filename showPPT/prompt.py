# from transformers import AutoTokenizer, AutoModel
# import os

# # 设置环境变量，启用离线模式
# os.environ['TRANSFORMERS_OFFLINE'] = '1'


# # 指定本地模型路径
# local_model_path = "/mnt/workspace/glm4/model"

# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(local_model_path, device_map='auto', trust_remote_code=True)
# model.eval()

# # 准备输入
# prompt = 

# # 使用模型生成文本
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_length=2000, do_sample=True, top_p=0.9)

# # 解码生成的文本
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f'ChatGLM：\n{generated_text}')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

query = "人为什么要学习？"

messages = [
    {
        "role": "system",
        "content": "对于我的问题，你要用大道至简的佛学来解答,不要超过200字"
    },
    {
        "role": "user",
        "content": query
    }
]

inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))