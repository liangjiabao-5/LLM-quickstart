from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from langchain.vectorstores import FAISS

# 读取内容
with open("real_estate_sales_data.txt") as f:
    real_estate_sales = f.read()

# 使用 CharacterTextSplitter 进行文本分割
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.create_documents([real_estate_sales])

# 模型路径，需要根据你的实际情况进行修改
MODEL_PATH = "/mnt/workspace/glm3/models"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, output_hidden_states=True).to('cuda').eval()

# 使用模型生成文本的嵌入向量
# 初始化一个空列表来存储嵌入向量
embeddings_list = []
texts = []

# 遍历文档列表
for doc in docs:
    # 提取文档的page_content属性
    text = doc.page_content
    
    # 使用分词器处理文本，并生成嵌入向量
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    with torch.no_grad():  # 禁用梯度计算，加快推理速度
        outputs = model(**inputs)
        # 获取所有层的hidden states
        hidden_states = outputs.hidden_states
        # 取最后一层的hidden states
        last_layer_hidden_states = hidden_states[-1]
        # 取最后一层的hidden states的平均值作为文档的嵌入向量
        embeddings = last_layer_hidden_states.mean(dim=1).cpu().numpy()
        
        # 将生成的嵌入向量添加到列表中
        embeddings_list.append(embeddings)
        texts.append(text)

# 使用预计算的嵌入向量创建 FAISS 索引
embedding_dim = embeddings_list[0].shape[1]  # 获取嵌入向量的维度
index = faiss.IndexFlatL2(embedding_dim)  # 创建一个空的L2索引
index.add(torch.tensor(embeddings_list).float())  # 将嵌入向量添加到索引中

# 创建向量数据库
db = FAISS.from_texts(texts=texts, embedding=embeddings_list)

# 查询文本
query = "小区吵不吵"

# 使用你的模型生成查询的嵌入向量
inputs = tokenizer(query, return_tensors="pt").to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
    query_vector = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()

# 使用查询向量进行相似性搜索
answer_list = db.similarity_search_by_vector(query_vector[0], k=5)  # 返回5个最相似的结果

# 输出结果
for ans in answer_list:
    print(ans.page_content + "\n")
