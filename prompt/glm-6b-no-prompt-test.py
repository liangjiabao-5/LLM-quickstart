from transformers import AutoTokenizer, AutoModel
import os

# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# 指定本地模型路径
local_model_path = "/mnt/workspace/glm3/models"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, device_map='auto', trust_remote_code=True)
model.eval()

# 准备输入
input_text = """对于我的问题，你要用大道至简的佛学来解答：
    人为什么要学习？
"""
print(f'输入：\n{input_text}')

response, history = model.chat(tokenizer=tokenizer, query=input_text)
print(f'ChatGLM3-6B：\n{response}')

# # 多轮对话
# while True:
#     query = input("\n用户：")
#     if query.strip() == "stop":
#         break
#     response, history = model.chat(tokenizer, query, history=[])
#     print("ChatGLM3：", response)