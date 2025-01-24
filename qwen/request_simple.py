import requests
import json

# API 的URL
url = "http://127.0.0.1:8000/generate/"

# 要发送到API的数据
data = {
    "text": "能给我推荐点北京的旅游景点吗！"  # 这里填入你想要生成文本的提示
}

# 设置请求头为 application/json
headers = {
    'Content-Type': 'application/json'
}

try:
    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析并打印返回的JSON数据
        result = response.json()
        print("Generated Text:", result.get('generated_text'))
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response body:", response.text)

except requests.exceptions.RequestException as e:
    # 处理请求异常
    print(f"An error occurred: {e}")