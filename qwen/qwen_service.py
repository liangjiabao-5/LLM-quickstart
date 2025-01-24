from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 Qwen 2.5 模型和分词器，并将其移动到 GPU（如果可用）
model_name = "/mnt/workspace/qwen/model-7B"  # 这里应该替换为实际的模型名称或路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype="auto",
                                             device_map="auto")

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class InputMessages(BaseModel):
    messages: list[Message]

@app.post("/generate/")
async def generate_text(input_data: InputMessages):
    print(input_data)
    # 查找用户输入的内容
    user_message = next((msg for msg in input_data.messages if msg.role == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing user message")
    
    # 查找系统提示的内容
    system_message = next((msg for msg in input_data.messages if msg.role == "system"), None)
    if not system_message:
        raise HTTPException(status_code=400, detail="Missing system prompt")
    
    print(f"{system_message.content}\n{user_message.content}")
    text = tokenizer.apply_chat_template(
        f"{system_message.content}\n{user_message.content}",
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 将输入文本编码为张量，并移动到 GPU（如果可用）
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 使用模型生成文本，并确保输出也在 GPU 上
    with torch.no_grad():  # 关闭梯度计算以节省内存
        outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1)
    
    # 解码生成的文本并返回
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)