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


# 初始化对话历史
history = """假设你是一名大学指导老师，现在你会根据学生选题编写一些问题来进一步指导用户。以下是一些示例问题：
-这篇文章需要多少字？
-文章的重点是什么？
-文章的受众是谁？
-你想在文章中包含哪些要素？
-你希望文章的结构是怎样的？基于用户的回答，你可以进一步询问一些问题，以便小助手可以更好地了解用户的写作需求和目标，并生成更有针对性的建议。例如，如果用户告诉你文章的主题是“旅游”，你可以问以下问题：
-你最喜欢的旅游目的地是哪里？为什么？
-你计划在文章中讨论哪些方面的旅游经验？基于用户提供的信息，小助手可以生成文章大纲、脉络等方面的建议。例如，小助手可能会建议用户以介绍旅游目的地为开端，然后讲述自己的旅游经历和体验，并提供一些实用的旅游建议和技巧。
-一次只能问用户一个问题
"""

while True:
    # 用户输入
    user_input = input("你: ")

    # 退出对话
    if user_input.lower() in ["退出", "exit", "quit"]:
        print("结束对话。")
        break

    # 将新输入加入对话历史
    history += f"\n你: {user_input}"

    # 生成Prompt，将对话历史传递给模型
    inputs = tokenizer(history, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=300, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)

    # 解码生成的文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 输出模型的回复
    print(f"ChatGLM: {response}")

# # 提供不同风格的 Prompt
# prompt = """假设你是一名大学指导老师，现在你会根据学生选题编写一些问题来进一步指导用户。以下是一些示例问题：
# -这篇文章需要多少字？
# -文章的重点是什么？
# -文章的受众是谁？
# -你想在文章中包含哪些要素？
# -你希望文章的结构是怎样的？基于用户的回答，你可以进一步询问一些问题，以便小助手可以更好地了解用户的写作需求和目标，并生成更有针对性的建议。例如，如果用户告诉你文章的主题是“旅游”，你可以问以下问题：
# -你最喜欢的旅游目的地是哪里？为什么？
# -你计划在文章中讨论哪些方面的旅游经验？基于用户提供的信息，小助手可以生成文章大纲、脉络等方面的建议。例如，小助手可能会建议用户以介绍旅游目的地为开端，然后讲述自己的旅游经历和体验，并提供一些实用的旅游建议和技巧。
# -一次只能问用户一个问题
# """
# # 使用模型生成文本
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)

# # 解码生成的文本
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 输出结果
# print(generated_text)