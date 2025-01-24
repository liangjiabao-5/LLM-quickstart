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
使用以下上下文，判断用户输入问题是否符合。
[内容]
1）经核查，机房所在建筑有抗震设防审批文档；
2）经核查，机房屋顶存在雨水渗透痕迹；
3）经核查，机房门窗已进行了密封，机房内不存在因风导致的尘土严重；
4）经核查，机房屋顶、四周墙体及地面无破损开裂。
[回答]
部分符合

[内容]
1）经核查，机房所在建筑有抗震设防审批文档；
2）经核查，机房天花板、窗台无水渗透现象；
3）经核查，机房门窗进行了密封，机房内不存在因风导致的尘土堆积现象，具有很好的防风能力；
4）现场观测，机房屋顶、四周墙体、门窗及地面无破损开裂的现象，具有很好的防震能力。
[回答]
符合

[内容]
1）经核查，机房所在建筑有抗震设防审批文档“建筑验收报告”；
2）经核查，机房屋顶和窗户不存在雨水渗透痕迹；
3）经核查，机房门窗已进行了密封，机房内不存在因风导致的尘土严重；
4）经核查，机房屋顶、四周墙体、门窗及地面无破损开裂现象
[回答]
符合


用户输入："1）经核查，机房所在建筑有抗震设防审批文档；
2）经核查，机房无窗户，天花板无水渗透现象；
3）经核查，机房门进行了密封，机房内不存在因风导致的尘土堆积现象，具有很好的防风能力；
4）经核查，机房屋顶、四周墙体、门及地面有破损开裂的现象，具有很好的防震能力。
"""

# 对Prompt进行编码
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成文本
output = model.generate(**inputs, max_length=2000, do_sample=True, top_p=0.9, temperature=0.7)

# 解码生成的文本
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)