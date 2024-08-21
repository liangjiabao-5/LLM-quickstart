import pandas as pd
import re

# 指定Excel文件路径
file_path = './metedata/安全物理环境.xlsx'

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
