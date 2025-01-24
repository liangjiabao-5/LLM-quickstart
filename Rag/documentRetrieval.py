# 更新导入路径
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings  # 更新导入路径
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os

# 加载PDF并进行文本分割
#loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")
loader = PyPDFLoader("https://www2.deloitte.com/content/dam/Deloitte/cn/Documents/technology/deloitte-cn-tech-state-of-ai-in-the-enterprise-2nd-edition-zh-190122.pdf")
pages = loader.load_and_split()

# 使用 RecursiveCharacterTextSplitter 进行文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_overlap=50,
    chunk_size=500
)
docs = text_splitter.split_documents(pages)

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

# 查询文本
query = "什么是企业高管们关心的主要人工智能风险？"

# 使用向量数据库进行查询
answer_list = db.similarity_search(query)

for ans in answer_list:
    print(ans.page_content + "\n")