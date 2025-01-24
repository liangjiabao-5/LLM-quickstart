import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


content1 = "经核查，机房内通信线路部署在机柜下方，分开铺设于不同的线槽中，部署位置隐蔽且安全，可以防止线缆受损。"

prompt1 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
预期结果：
机房通信线缆铺设在线槽或桥架里。

最后判断，如果预期结果符合content的内容，则输出"结果：符合"，如果不符合，则输出"结果：不符合"
"""

content2 = "经核查，机房内采取了措施防止静电的产生，设置了防静电手环等。"

prompt2 = """
按照以下预期结果的规则，判断content是否符合:
预期结果：
机房内配备了静电消除设备。

最后判断，如果预期结果符合content的内容，则输出"结果：符合"，如果不符合，则输出"结果：不符合"
"""


inputs = tokenizer.apply_chat_template([{"role": "system", "content": prompt2 },{"role": "user", "content": content2}],
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