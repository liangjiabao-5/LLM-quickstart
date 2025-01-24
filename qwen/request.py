import requests
import json

# API 的URL
url = "http://127.0.0.1:8000/generate/"

# 输入的内容
content = "1.经核查，机房及相关的工作房间和辅助房采用了隔热防火门、墙壁涂抹了防护涂料等具有耐火等级的建筑材料；2.经核查，机房建设时具有防火验收材料。"

# 系统提示语
prompt = """
按照以下预期结果的几点规则，判断user输入的content是否符合:
预期结果：
机房使用的所有材料为耐火材料，例如使用墙体、防火玻璃等，但使用金属栅栏的情况不能算符合。

最后判断，如果每条预期结果都符合则输出"结果：符合"，部分预期结果符合输出"结果：部分符合"，每条预期结果都不符合则输出"结果：不符合"
"""

# 要发送到API的数据
messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": content}
]

# 设置请求头为 application/json
headers = {
    'Content-Type': 'application/json'
}

try:
    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps({"messages": messages}))

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析返回的JSON数据
            result = response.json()
            generated_text = result.get('generated_text', 'No generated text found')
            print("Generated Text:", generated_text)
        except ValueError:
            print("Error: Response is not a valid JSON")
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response body:", response.text)

except requests.exceptions.RequestException as e:
    # 处理请求异常
    print(f"An error occurred: {e}")