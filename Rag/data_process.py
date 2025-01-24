import pandas as pd

# 读取Excel文件
df = pd.read_excel('report_db2_content_results_info.xls')

# 打开一个文件用于写入
with open('report_data_output.txt', 'w', encoding='utf-8') as file:
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        context = row['context']
        answer = row['answer']
        
        # 写入内容和回答到文件
        file.write(f"{index + 1}.\n[内容]\n")
        file.write(context + "\n")
        file.write("[回答]\n")
        file.write(answer + "\n\n")  # 添加两个换行符作为分隔