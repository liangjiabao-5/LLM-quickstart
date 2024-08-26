from transformers import AutoTokenizer, AutoModel
import os

# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 设置本地缓存目录
os.environ['HF_HUB_CACHE'] = '/mnt/new_volume/hf/hub'

# 指定本地模型路径
local_model_path = os.path.join(os.environ['HF_HUB_CACHE'], 'chatglm3-6b')

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, device_map='auto', trust_remote_code=True)
model.eval()

# 准备输入
input_text = """对于我的问题，你要用大道至简的佛学或中国哲学来解答：
    人为什么要上班？
"""
print(f'输入：\n{input_text}')

response, history = model.chat(tokenizer=tokenizer, query=input_text)
print(f'ChatGLM3-6B：\n{response}')

