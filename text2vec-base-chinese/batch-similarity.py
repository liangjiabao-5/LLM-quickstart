import torch
from transformers import AutoTokenizer, AutoModel
import os
from scipy.spatial.distance import cosine

# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 设置本地缓存目录
os.environ['HF_HOME'] = '/mnt/new_volume/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/new_volume/hf/hub'

# 指定本地模型路径
local_model_path = os.path.join(os.environ['HF_HUB_CACHE'], 'text2vec-base-chinese')

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

# 准备文本对
text_pairs = [
    ("物理位置选择", "核查是否有建筑物抗震设防审批文档"),
    ("物理位置选择", "核查是否有可灵活开启的窗户，若有窗户，则核查是否做了封闭、上锁等防护措施；"),
    ("物理位置选择", "机房场地应选择在具有防震、防风和防雨等能力的建筑内"),
    ("物理访问控制", "核查出入口是否配置电子门禁系统"),
]

# 批量计算文本对的相似度
for text1, text2 in text_pairs:
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # 生成向量
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 计算余弦相似度
    text_vector1 = outputs1.last_hidden_state.mean(dim=1)
    text_vector2 = outputs2.last_hidden_state.mean(dim=1)

    similarity = 1 - cosine(text_vector1[0].numpy(), text_vector2[0].numpy())
    print(f"文本相似度（{text1} vs {text2}）: {similarity}")

