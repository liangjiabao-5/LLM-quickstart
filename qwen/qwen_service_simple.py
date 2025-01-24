from fastapi import FastAPI
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

class InputText(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(input_data: InputText):
    # 将输入文本编码为张量，并移动到 GPU（如果可用）
    inputs = tokenizer(input_data.text, return_tensors="pt").to(model.device)
    
    # 使用模型生成文本，并确保输出也在 GPU 上
    with torch.no_grad():  # 关闭梯度计算以节省内存
        outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1)
    
    # 解码生成的文本并返回
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)