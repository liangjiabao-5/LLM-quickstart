from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/workspace/qwen/model"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

content = "设备和系统部署在生产控制大区安全区Ⅰ，与安全区Ⅱ、管理信息大区之间无数据交互，物理隔离，内部网络环境可控；同时电力企业在物理环境防护上制定了严格的管理制度《息网络运行维护管理制度》进行管控，有效降低了漏洞获取和利用的安全风险。"

prompt = """
# Role:文本主题提取器

## Goals
- 对给定文本提取主题，并仅输出相应的主题内容和按主题分割后的句子。

## Constrains
- 从以下主题中为文本匹配相关主题词，可以是一个或多个主题词：本地运维、视频监控、网络防护、内网隔离、管理制度、局域网运维。
- 输出结果必须仅为主题名称和主题句，不能包含其他多余信息。

## Skills
- 强化文本摘要的能力
- 理解并解析文本内容
- 确定文本包含的主题

## outfromt
- 输出格式: [主题名称,主题句]

## Workflow
1. 读取并理解给定的文本。
2. 根据文本内容，判断其包含的主题。
3. 输出判断出的主题名称和主题句。
"""

messages = [
    {"role": "system", "content": prompt },
    {"role": "user", "content": content }
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)