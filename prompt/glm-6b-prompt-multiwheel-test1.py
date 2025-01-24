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

# 初始化对话上下文
conversation_history = [
    "你是一位精通佛学和中国哲学的AI助手，将用简洁的语言回答用户的问题。",
    "用户: 你是谁",
    "AI: 我是一位精通佛学和中国哲学的AI助手，将用简洁的语言回答用户的问题。"
]

def truncate_history(history, max_length=1000):
    """确保对话历史不超过一定长度"""
    while len("\n".join(history)) > max_length:
        history.pop(0)

# 进行多轮对话
while True:
    try:
        # 获取用户输入
        user_input = input("用户: ")

        # 如果用户输入为空，跳过本次循环
        if not user_input.strip():
            continue

        # 将用户输入添加到对话历史中
        conversation_history.append(f"用户: {user_input}")
        conversation_history.append("AI: ")

        # 修剪对话历史以控制长度
        truncate_history(conversation_history, max_length=1500)

        # 对Prompt进行编码
        inputs = tokenizer("\n".join(conversation_history), return_tensors="pt").to("cuda")

        # 生成AI的回答
        output = model.generate(**inputs, max_length=200, do_sample=True, top_p=0.9, temperature=0.7)

        # 解码生成的文本
        ai_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # 打印AI的回答
        print(f"AI: {ai_response}")

        # 将AI的回答添加到对话历史中
        conversation_history[-1] += ai_response

    except Exception as e:
        print(f"发生错误: {str(e)}")
        continue