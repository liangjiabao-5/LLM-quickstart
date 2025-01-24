import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

query = """1）经核查，机房所在建筑有抗震设防审批文档；
2）经核查，机房无窗户，天花板存在水渗透现象；
3）经核查，机房门进行了密封，机房内不存在因风导致的尘土堆积现象，具有很好的防风能力；
4）经核查，机房屋顶、四周墙体、门及地面有破损开裂的现象。
"""

messages = [
    {
        "role": "system",
        "content": """
        按照以下预期结果的几点规则，判断用户输入的内容是否符合:
        预期结果：
        1)机房具有验收文档；
        2)天花板、窗台无水渗漏现象；
        3)机房无窗户，或者有窗户且采取了防护措施；
        4)现场观测屋顶、墙体、门窗和地面等，无开裂现象。
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