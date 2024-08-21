import pandas as pd
import re
import dashscope
import csv

# 声明一个空的列表来存储新的数据行
new_data = []

# 指定Excel文件路径
file_path = './metedata/安全管理人员.xlsx'

# 读取Excel文件
df = pd.read_excel(file_path)

# 如果不知道列名，直接通过索引获取
df.iloc[:, 1] = df.iloc[:, 1].ffill()
df.iloc[:, 0] = df.iloc[:, 0].ffill()
# 行，列
data = df.iloc[1:, 2]

# 输出读取的数据
for index, value in enumerate(data):
        #left_value = df.iloc[index + 2, 1]  # 注意索引偏移
        #print(f"Row {index + 3}: Column Value: {value}, Left Column Value: {left_value}")
        question_text = f"\“在进行等级保护测评时，针对{df.iloc[index + 1, 0]}大类，{df.iloc[index + 1, 1]}的安全控制点下，关于{value}的测评指标。需要进行哪些操作步骤？\”请根据这句话给出20种可能提问的方式，用来大模型训练";
        question = re.sub(r'\(G3\)|a\)|b\)|c\)|e\)|f\)|d\)|\(S3\)|\。|\；', '', question_text)
        print(question)
        operation_step = df.iloc[index + 1, 3]
        print(operation_step)

        # 如果环境变量配置无效请启用以下代码
        # dashscope.api_key = 'YOUR_DASHSCOPE_API_KEY'


                #messages = [{'role': 'user', 'content': '如何做炒西红柿鸡蛋？'}]
        messages=[
                    {'role': 'system', 'content': "你是一个网络安全等级保护测评专家，尤其擅长对信息安全技术网络等级保护测评要求的解读"},
                        {'role': 'user', 'content': question}
        ]

        response = dashscope.Generation.call(dashscope.Generation.Models.qwen_turbo,messages=messages,result_format='message')
        print(response)
        # 将 JSON 字符串转换为字典
        #data_dict = json.loads(response)


        # 提取 'content' 和 'summary'
        text = response['output']['choices'][0]['message']['content']

        print("Content:", text)

        # 分割字符串为单独的问题
        questions = text.split('\n')

        # 每个问题的summary（这里假设所有的summary都是一样的）
        summary = operation_step

        # 写入数据
        for q in questions:
            # 去除每个问题前面的序号
            #content = q.split('.', 1)[1].strip()
            # 检查分割后的列表长度是否足够
            parts = q.split('.', 1)
            if len(parts) > 1:
                content = parts[1].strip()
                new_data.append((content, summary))
        print("--------------------------------------------------------------------------------------------------------------")

# 创建CSV文件并写入数据
with open('Safety_management_personnel_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['content', 'summary'])
    # 遍历new_data并写入每一行
    for row in new_data:
        writer.writerow(row)
