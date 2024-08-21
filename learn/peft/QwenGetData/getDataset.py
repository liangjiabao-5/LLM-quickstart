import dashscope
import csv

# 如果环境变量配置无效请启用以下代码
# dashscope.api_key = 'YOUR_DASHSCOPE_API_KEY'


raw_content = "\“在进行等级保护测评时，针对安全物理环境大类，物理位置选择的安全控制点下，关于机房场地选择的测评指标，为了满足机房场地应选择在具有防震、防风和防雨等能力的建筑内的测评指标。需要进行哪些操作步骤？\”请根据这句话给出20种可能提问的方式，用来大模型训练"


#messages = [{'role': 'user', 'content': '如何做炒西红柿鸡蛋？'}]
messages=[
    {'role': 'system', 'content': "你是一个网络安全等级保护测评专家，尤其擅长对信息安全技术网络等级保护测评要求的解读"},
    {'role': 'user', 'content': raw_content}
]

response = dashscope.Generation.call(dashscope.Generation.Models.qwen_turbo,messages=messages,result_format='message')
print(response)
print("--------------------------------------------------------------------------------------------------------------")
# 将 JSON 字符串转换为字典
#data_dict = json.loads(response)


# 提取 'content' 和 'summary'
text = response['output']['choices'][0]['message']['content']

print("Content:", text)

# 分割字符串为单独的问题
questions = text.split('\n')

# 每个问题的summary（这里假设所有的summary都是一样的）
summary = """
1)核查是否有建筑物抗震设防审批文档；
2)核查是否有雨水渗漏的痕迹；
3)核查是否有可灵活开启的窗户，若有窗户，则核查是否做了封闭、上锁等防护措施；
4)核查屋顶、墙体、门窗和地面等是否有破损开裂的情况。
"""

# 创建CSV文件并写入数据
with open('zhouyi_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['content', 'summary'])

    # 写入数据
    for question in questions:
        # 去除每个问题前面的序号
        content = question.split('.', 1)[1].strip()
        writer.writerow([content, summary])

