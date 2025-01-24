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
你将获得由用户提供的双引号包裹的文本，按照以下分步骤,处理文本内容:
1 -用一个以"总结:"为前缀的句子,对文本进行简单总结。
2 -将1中总结的内容翻译成英语,并在前面加上"English:"作为前缀。
3 -将1中总结的内容翻译成日语,并在前面加上"まとめ:“作为前缀。

用户输入："容器之所以广受欢迎，是因为它能简化应用或服务及其所有依赖项的构建、封装与推进，而且这种简化涵盖整个生命周期，跨越不同的工作流和部署目标。然而，容器安全依然面临着一些挑战。虽然容器有一些固有的安全优势（包括增强的应用隔离），但也扩大了企业的威胁范围。如果不能识别和规划与容器相关的特定安全措施，可能会增加企业的安全风险"
"""

# 对Prompt进行编码
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成文本
output = model.generate(**inputs, max_length=2000, do_sample=True, top_p=0.9, temperature=0.7)

# 解码生成的文本
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)