from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/workspace/qwen/model-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

content1 = "经核查，机房处在麻岭330kV变电站一层，共一层，机房加强了防水和防潮措施；机房无窗，采用了门密封、部署专用排水管道、空调除湿等防水和防潮措施。"

prompt1 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
预期结果：
1)机房未设置在建筑物的顶层或地下室；
2)设置在建筑物的顶层或地下室的机房，采取了严格的防水和防潮措施。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合
"""

content2 = "1.经核查，机房及相关的工作房间和辅助房采用了隔热防火门、墙壁涂抹了防护涂料等具有耐火等级的建筑材料；2.经核查，机房建设时具有防火验收材料。"

prompt2 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
预期结果：
机房使用的所有材料为耐火材料，例如使用墙体、防火玻璃等，但使用金属栅栏的情况不能算符合。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合
"""

messages = [
    {"role": "system", "content": prompt2 },
    {"role": "user", "content": content2 }
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