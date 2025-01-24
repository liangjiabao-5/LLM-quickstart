# 更新导入路径
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings  # 更新导入路径
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os

from langchain.text_splitter import CharacterTextSplitter

# 读取内容
with open("real_estate_sales_data.txt") as f:
    real_estate_sales = f.read()

# 使用 CharacterTextSplitter 进行文本分割
text_splitter = CharacterTextSplitter(
    separator = r'\d+\.',
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = True,
)
docs = text_splitter.create_documents([real_estate_sales])

# 提取文本内容
texts = [doc.page_content for doc in docs]

# 使用SentenceTransformer模型
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', '/mnt/workspace/glm3/m3')
embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")

# 使用模型生成文本的嵌入向量
embeddings = embedding_model.encode(texts, convert_to_tensor=True)

# 创建Chroma数据库并嵌入文档
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_PATH)  # 移除device参数
db = Chroma.from_documents(documents=docs, embedding=embedding_function, collection_name="embed")


# query = "小区吵不吵"
# answer_list = db.similarity_search(query)
# for ans in answer_list:
#     print(ans.page_content + "\n")

    
#-----------------------------------------

# 查询文本
query = "小区吵不吵"

# 使用向量数据库进行查询，提取前三个
topK_retriever = db.as_retriever(search_kwargs={"k": 3})
results = topK_retriever.get_relevant_documents(query)
print("使用向量数据库进行查询，提取前三个相近的：-------------------------------------")
for result in results:
    print(result.page_content + "\n")

    
#-----------------------------------------
#经过测试没有任何效果
# from typing import List

# def sales(query: str, score_threshold: float=0.8) -> List[str]:
#     # 实例化一个 similarity_score_threshold Retriever,提升结果的相关性质量
#     retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold})    
#     docs = retriever.get_relevant_documents(query)
#     ans_list = [doc.page_content.split("[销售回答] ")[-1] for doc in docs]

#     return ans_list

# query = "小区吵不吵"

# print(f"score:0.8 ans: {sales(query)}\n")
# print(f"score:0.75 ans: {sales(query, 0.75)}\n")
# print(f"score:0.5 ans: {sales(query, 0.5)}\n")
# print(f"score:0.3 ans: {sales(query, 0.3)}\n")
# print(f"score:0.1 ans: {sales(query, 0.1)}\n")
# print(f"score:0.9 ans: {sales(query, 0.9)}\n")


def augment_prompt(query: str):
    # 获取top3的文本片段
    source_knowledge = "\n".join([x.page_content for x in results])
    # 构建prompt
    augmented_prompt = f"""使用以下上下文，回答问题。
    
内容：
{source_knowledge}
    
问题：
{query}"""
    return augmented_prompt

prompt = augment_prompt(query)
print("模型prompt修饰内容如下：-------------------------------------")
print(prompt)

# -------------------调用模型进行生成------------------------
from transformers import AutoTokenizer, AutoModel

# 设置环境变量，启用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# 指定本地模型路径
local_model_path = "/mnt/workspace/glm3/models"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, device_map='auto', trust_remote_code=True)
model.eval()

# 使用模型生成文本
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=1000, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 输出结果
print("模型输出结果：-------------------------------------")
print(generated_text)