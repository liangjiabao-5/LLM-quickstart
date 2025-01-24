import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


query = "容器之所以广受欢迎，是因为它能简化应用或服务及其所有依赖项的构建、封装与推进，而且这种简化涵盖整个生命周期，跨越不同的工作流和部署目标。"

messages = [
    {
        "role": "system",
        "content": """
        你将获得由用户提供的文本，按照以下分步骤,处理文本内容:
        1 -用一个以"总结:"为前缀的句子,对文本进行简单总结。
        2 -将1中总结的内容翻译成英语,并在前面加上"English:"作为前缀。
        3 -将1中总结的内容翻译成日语,并在前面加上"まとめ:“作为前缀。
        """
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