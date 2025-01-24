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
# 提供不同风格的 Prompt
prompt = """
你将获得由用户提供的文本，该文本应为问题的答案。检查以下信息是否直接包含在答案中：
-尼尔阿姆斯特朗是第一个登上月球的人。
-尼尔阿姆斯特朗首次登上月球的日期是1969年7月21日。

对于每个要点，请执行以下步骤：

1 -陈述该要求。
2 -提供与此要点最接近的答案引用。
3 -考虑一下，如果不了解这个主题的读者阅读引用，能否直接推断出该要点。在做出决定之前，请解释原因。
4 -如果问题3的答案是"是"，则写入"yes"，否则写入"no"。

最后，提供有多少个"yes"回答。将此计数作为{"count":<insert count here>}。

用户：尼尔阿姆斯特朗因为成为第一个登上月球的人而闻名。这一历史性事件发生在1967年7月21日，当时是阿波罗11号任务。
"""

# 使用模型生成文本
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=1000, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 输出结果
print(generated_text)