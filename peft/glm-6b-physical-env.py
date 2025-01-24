from datasets import load_dataset
import os

# 定义全局变量和参数
model_name_or_path = 'THUDM/chatglm3-6b'
train_data_path = 'dataset/Secure_physical_environment_dataset.csv'    # 训练数据路径
book = 'Secure_physical_environment'
eval_data_path = None                     # 验证数据路径，如果没有则设置为None
seed = 8                                 # 随机种子
max_input_length = 512                    # 输入的最大长度
max_output_length = 1536                  # 输出的最大长度
lora_rank = 4                             # LoRA秩
lora_alpha = 32                           # LoRA alpha值
lora_dropout = 0.05                       # LoRA Dropout率
prompt_text = ''                          # 所有数据前的指令文本


# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 设置本地缓存目录
os.environ['HF_DATASET_CACHE'] = '/mnt/workspace/quickstart/learn/peft/dataset'
os.environ['HF_HUB_CACHE'] = '/mnt/new_volume/hf/hub'

# 指定本地模型路径
local_dataset_path = os.path.join(os.environ['HF_DATASET_CACHE'], 'adgen')
local_model_path = os.path.join(os.environ['HF_HUB_CACHE'], 'chatglm3-6b')

# -----------数据集处理

print("------------准备数据集----------------")
dataset = load_dataset("csv", data_files=train_data_path)

from transformers import AutoTokenizer

# revision='b098244' 版本对应的 ChatGLM3-6B 设置 use_reentrant=False
# 最新版本 use_reentrant 被设置为 True，会增加不必要的显存开销
tokenizer = AutoTokenizer.from_pretrained(local_model_path,
                                          trust_remote_code=True,
                                          revision='b098244')

# tokenize_func 函数
def tokenize_func(example, tokenizer, ignore_label_id=-100):
    """
    对单个数据样本进行tokenize处理。

    参数:
    example (dict): 包含'content'和'summary'键的字典，代表训练数据的一个样本。
    tokenizer (transformers.PreTrainedTokenizer): 用于tokenize文本的tokenizer。
    ignore_label_id (int, optional): 在label中用于填充的忽略ID，默认为-100。

    返回:
    dict: 包含'tokenized_input_ids'和'labels'的字典，用于模型训练。
    """

    # 构建问题文本
    question = prompt_text + example['content']
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    # 构建答案文本
    answer = example['summary']

    # 对问题和答案文本进行tokenize处理
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    # 如果tokenize后的长度超过最大长度限制，则进行截断
    if len(q_ids) > max_input_length - 2:  # 保留空间给gmask和bos标记
        q_ids = q_ids[:max_input_length - 2]
    if len(a_ids) > max_output_length - 1:  # 保留空间给eos标记
        a_ids = a_ids[:max_output_length - 1]

    # 构建模型的输入格式
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # 加上gmask和bos标记

    # 构建标签，对于问题部分的输入使用ignore_label_id进行填充
    labels = [ignore_label_id] * question_length + input_ids[question_length:]

    return {'input_ids': input_ids, 'labels': labels}

column_names = dataset['train'].column_names
tokenized_dataset = dataset['train'].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False, 
    remove_columns=column_names
)
# show_random_elements(tokenized_dataset, num_examples=1)

tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
tokenized_dataset = tokenized_dataset.flatten_indices()

import torch
from typing import List, Dict, Optional

# DataCollatorForChatGLM 类
class DataCollatorForChatGLM:
    """
    用于处理批量数据的DataCollator，尤其是在使用 ChatGLM 模型时。

    该类负责将多个数据样本（tokenized input）合并为一个批量，并在必要时进行填充(padding)。

    属性:
    pad_token_id (int): 用于填充(padding)的token ID。
    max_length (int): 单个批量数据的最大长度限制。
    ignore_label_id (int): 在标签中用于填充的ID。
    """

    def __init__(self, pad_token_id: int, max_length: int = 2048, ignore_label_id: int = -100):
        """
        初始化DataCollator。

        参数:
        pad_token_id (int): 用于填充(padding)的token ID。
        max_length (int): 单个批量数据的最大长度限制。
        ignore_label_id (int): 在标签中用于填充的ID，默认为-100。
        """
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """
        处理批量数据。

        参数:
        batch_data (List[Dict[str, List]]): 包含多个样本的字典列表。

        返回:
        Dict[str, torch.Tensor]: 包含处理后的批量数据的字典。
        """
        # 计算批量中每个样本的长度
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)  # 找到最长的样本长度

        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d  # 计算需要填充的长度
            # 添加填充，并确保数据长度不超过最大长度限制
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        # 将处理后的数据堆叠成一个tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return {'input_ids': input_ids, 'labels': labels}
    
data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)
print("------------数据集准备完毕----------------")
# ---------------------

from transformers import AutoModel, BitsAndBytesConfig

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

# QLoRA 量化配置
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

# 加载模型
print("------------加载模型----------")
model = AutoModel.from_pretrained(local_model_path,
                                  quantization_config=q_config,
                                  device_map='auto',
                                  trust_remote_code=True,
                                  revision='b098244')
# 这行代码设置模型以支持梯度检查点（gradient checkpointing），这是一种节省内存的技术，可以在训练期间减少内存使用量。
model.supports_gradient_checkpointing = True
# 启用梯度检查点。当启用时，模型在前向传播期间会丢弃一些中间激活，然后在反向传播时重新计算它们。
model.gradient_checkpointing_enable()
# 这告诉模型输入需要计算梯度，这对于梯度下降和其他需要计算梯度的优化算法是必要的。
model.enable_input_require_grads()
# 这行代码禁用了模型配置中的缓存使用。在训练期间，Transformers库可能会缓存一些中间结果以加速后续的相同输入。然而，在某些情况下，这可能会导致警告信息。通过设置use_cache为False，可以禁用这种行为，从而避免这些警告。注释中提到的“Please re-enable for inference!”意味着在推理（即模型部署和使用）阶段，应该重新启用缓存以提高效率。
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# 预处理量化模型 
from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training

kbit_model = prepare_model_for_kbit_training(model)

# LoRA 适配器配置 
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)

qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()


import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

train_epochs = 3
output_dir = f"models/{book}/{model_name_or_path}-epoch{train_epochs}-{timestamp}"

# 训练超参数配置
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=output_dir,                            # 输出目录
    per_device_train_batch_size=8,                     # 每个设备的训练批量大小
    gradient_accumulation_steps=4,                     # 梯度累积步数
    learning_rate=1e-3,                                # 学习率
    num_train_epochs=train_epochs,                     # 训练轮数
    lr_scheduler_type="linear",                        # 学习率调度器类型
    warmup_ratio=0.1,                                  # 预热比例
    logging_steps=1,                                 # 日志记录步数
    save_strategy="steps",                             # 模型保存策略
    save_steps=10,                                    # 模型保存步数
    optim="adamw_torch",                               # 优化器类型
    fp16=True,                                        # 是否使用混合精度训练
)

# 训练参数
trainer = Trainer(
        model=qlora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

print("------------准备训练----------------")
trainer.train()

print("------------准备保存模型----------------")
trainer.model.save_pretrained(output_dir)