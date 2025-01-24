from transformers import AutoTokenizer, AutoModel
import os

# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'



# 指定本地模型路径
local_model_path = "/mnt/workspace/glm3/models"
# local_model_path = os.path.join(os.environ['HF_HUB_CACHE'], 'model')

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, device_map='cuda', trust_remote_code=True)
model.eval()


# 设置Prompt
prompt = """
按照以下预期结果的几点规则，判断用户输入的内容是否符合，如果符合请输出"符合"，如果不符合请输出原因:
预期结果：
1)机房具有验收文档；
2)天花板、窗台无水渗漏现象；
3)机房无窗户，或者有窗户且采取了防护措施；
4)现场观测屋顶、墙体、门窗和地面等，无开裂现象。


用户输入："机房所在建筑有抗震设防审批文档,屋顶存在雨水渗透痕迹,门窗已进行了密封，机房内不存在因风导致的尘土严重,屋顶、四周墙体及地面有破损开裂。"
"""

# 对Prompt进行编码
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成文本
output = model.generate(**inputs, max_length=2000, do_sample=True, top_p=0.9, temperature=0.7)

# 解码生成的文本
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)