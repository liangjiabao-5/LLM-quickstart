from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 物理位置选择
content1 = "1）经核查，机房所在建筑具备抗震设防审批文档，相关文档：“抗震设防审批文件”；2）经核查，机房屋顶和窗户不存在雨水渗透痕迹；3）经核查，机房门窗已进行了密封，机房内不存在因风导致的尘土；4）经核查，机房屋顶、四周墙体、门窗及地面无破损开裂。"

prompt1 = """
按照以下的操作步骤进行判断，user输入的content是否符合:
请注意：
1.在评估时对具体对象进行比对，识别和理解它们所属的类别或具有的属性一致则为符合。例如："抗震设防审批文件"属于"验收文档"
2.预期结果中包含互斥情况。当user输入的content明确支持预期结果中的某一情况时，模型应专注于评估该情况，而无需对其他互斥情况进行判断。
3.只要语义相同即可，进行模糊匹配。

预期结果：
1)机房具有验收文档；
2)天花板、窗台无水渗漏现象；
3)机房无窗户，或者有窗户且采取了防护措施；
4)现场观测屋顶、墙体、门窗和地面等，无开裂现象。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合"。
"""

# 防火
content2 = "经核查，机房内采取了措施防止静电的产生，设置了防静电手环等。"

prompt2 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
请注意：
1.在判断对具体对象进行比对时，识别和理解它们所属的类别或具有的属性相关则为符合。

预期结果：
机房内配备了静电消除设备。

最后判断，如果预期结果符合content其中的一条内容，则输出"结果：符合"，如果没有任何一条内容符合，则输出"结果：不符合"
"""
# TODO:注意逻辑
content3 = "经核查，机房出入口设置了海康威视的电子门禁系统，能够识别、记录进入人员信息。"

prompt3 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
请注意：
1.在评估时对具体对象进行比对，识别和理解它们所属的类别或具有的属性一致则为符合。
2.预期结果中包含互斥情况。当user输入的content明确支持预期结果中的某一情况时，模型应专注于评估该情况，而无需对其他互斥情况进行判断。

预期结果：
1)核查出入口是否配置电子门禁系统；
2)核查电子门禁系统是否开启并正常运行；
3)核查电子门禁系统是否可以鉴别、记录进入的人员信息；
4）无电子门禁时，查看是否有机械门锁，有无专人值守。


最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合
"""


# 物理位置选择
content4 = "经核查，机房配备了防静电手环，能够有效防止静电的产生。"

prompt4 = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
请注意：
1.在评估时对具体对象进行比对，识别和理解它们所属的类别或具有的属性一致则为符合。
2.只要语义相同即可，进行模糊匹配。

预期结果：
机房内配备了静电消除设备。

最后判断，如果content其中的一条内容符合预期结果，则输出"结果：符合"，如果没有任何一条内容符合，则输出"结果：不符合"。
"""

chat_response = client.chat.completions.create(
    model="/mnt/workspace/qwen/model-7B",
    messages=[
        {"role": "system", "content": prompt4},
        {"role": "user", "content": content4},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)

first_choice = chat_response.choices[0]

# 获取 Choice 对象的 message 属性
message = first_choice.message

# 获取 message 的 content 字段内容
content = message.content
print(f"获取qwen执行结果：{content}")
