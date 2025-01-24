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
model = AutoModel.from_pretrained(local_model_path, device_map='cuda', trust_remote_code=True)
model.eval()


# 设置Prompt
prompt = """
对于我的问题，你要用大道至简的佛学或中国哲学来解答，请用四字成语表达，更有大师范儿

用户输入: "为什么要天天上班"
你的回答:
"""

# 对Prompt进行编码
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成文本
output = model.generate(**inputs, max_length=1000, do_sample=True, top_p=0.9, temperature=0.7)

# 解码生成的文本
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)