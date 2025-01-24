import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

query = """1）经核查，机房配备了UPS系统，厂家：四川帝威能源技术有限公司，UPS系统蓄电池组容量为：500Ah，配有两组；
2）经核查，UPS系统蓄电池组有测试记录，UPS系统可以保证断电情况下4小时的正常运行。"""

role = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
请注意：
1.在评估时对具体对象进行比对，识别和理解它们所属的类别或具有的属性一致则为符合。
2.预期结果中包含互斥情况。当user输入的content明确支持预期结果中的某一情况时，模型应专注于评估该情况，而无需对其他互斥情况进行判断。
3.只要语义相同即可，进行模糊匹配。
                
预期结果：
1)机房配备了UPS系统；
2）UPS系统能够满足短期断电时的供电要求；
3）记录UPS数量以及厂家、型号。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合
"""

inputs = tokenizer.apply_chat_template([{"role": role, "content": query}],
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