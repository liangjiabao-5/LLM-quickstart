import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 设置本地缓存目录
os.environ['HF_HOME'] = '/mnt/new_volume/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/new_volume/hf/hub'

local_model_path = os.path.join(os.environ['HF_HUB_CACHE'], 'text2vec-base-chinese')


# 1. 加载CSV文件
df = pd.read_csv('../../dataset2.csv')

# 2. 将DataFrame转换为datasets.Dataset对象，并重命名列名以确保一致性
dataset = Dataset.from_pandas(df)

# 3. 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=2)

# 4. 定义数据预处理函数
def preprocess_function(examples):
    inputs = tokenizer(examples['text1'], examples['text2'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = examples['label']
    return inputs

# 5. 应用数据预处理函数
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 7. 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

# 8. 开始训练
trainer.train()

#保存模型和分词器
save_directory = "./resultsModel"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)