# main.py
from argparse import ArgumentParser
import os
from typing import List, Tuple

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

DEFAULT_CKPT_PATH = './model/qwen/Qwen-1_8B-Chat'

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    history: List[Tuple[str, str]]

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    args = parser.parse_args()
    return args

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, config
    args = _get_args()
    model, tokenizer, config = _load_model_tokenizer(args)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    _query = request.query
    _history = request.history
    full_response = ""

    for response in model.chat_stream(tokenizer, _query, history=_history, generation_config=config):
        full_response = response

    _history.append((_query, full_response))
    return {"response": full_response, "history": _history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)