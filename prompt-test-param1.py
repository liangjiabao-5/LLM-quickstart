import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "/mnt/workspace/glm4/model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

content = "1.经核查，机房及相关的工作房间和辅助房采用了隔热防火门、墙壁涂抹了防护涂料等具有耐火等级的建筑材料；2.经核查，机房建设时具有防火验收材料。"

prompt = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
预期结果：
机房使用的所有材料为耐火材料，例如使用墙体、防火玻璃等，但使用金属栅栏的情况不能算符合。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合"
"""

content1 = "1）经核查，机房内所有机柜、设施和设备有接地线进行接地处理；2）经核查，机柜内设备接地线连接了机柜。"

prompt1 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
预期结果：
机房内所有机柜、设施和设备等均已采取接地的控制措施。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合"
"""

content2 = "1)经核查，机房屋顶、墙壁采取防水涂层措施防止雨水渗透；2)经核查机房内不存在雨水渗透痕迹。"

prompt2 = """
按照以下预期结果的规则，判断content是否符合:
预期结果：
机房采取了防雨水渗透的措施，如封锁了窗户并采取了防水措施、屋顶和墙壁均采取了防雨水渗透的措施。

最后判断，如果预期结果符合content其中的一个内容，则输出"结果：符合"，如果没有任何内容符合，则输出"结果：不符合"
"""


inputs = tokenizer.apply_chat_template([{"role": "system", "content": prompt },{"role": "user", "content": content2}],
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