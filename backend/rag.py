import os
from functools import lru_cache

import redis
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model import embeddings_model

load_dotenv()  # 加载 .env 文件中的环境变量


def add_rag_pdf(file_path=r"rag\9787502958572_L.pdf"):
    """加载 PDF 并分块写入 Redis 向量库。

    file_path 缺省时使用项目内置的气象资料 PDF。
    """
    # load() 为 BaseLoader 统一接口，返回 List[Document]
    documents = PyMuPDFLoader(
        file_path=file_path,
        mode="page",  # plain 纯文本；layout 按版面
    ).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )

    splitter_documents = text_splitter.split_documents(documents)

    embeddings = embeddings_model()

    # 用新库 langchain_redis 批量写入并建索引(由于本地环境限制，批量写入时可能会报错，故而逐个写入)
    re = RedisVectorStore.from_documents(
        documents=[splitter_documents[0]],
        embedding=embeddings,
        config=RedisConfig(
            index_name="weather",
            redis_url=os.getenv("REDIS_URL"),
        ),
    )
    for i in splitter_documents[1:]:
        re.add_documents([i])


def del_rag(index_name):
    # 直接通过 redis 客户端删除索引及关联文档
    client = redis.from_url(os.getenv("REDIS_URL"))
    client.ft(index_name).dropindex(delete_documents=True)


@lru_cache(maxsize=1)
def get_vector_store():
    """获取缓存的 Redis 向量库客户端，避免每次检索都重建连接。"""
    embeddings = embeddings_model()
    config = RedisConfig(
        index_name="weather",
        redis_url=os.getenv("REDIS_URL"),
    )
    return RedisVectorStore(embeddings, config=config)


def get_retriever(question):
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, 5)
    result = ""
    for i, j in results:
        if j > 0.5:
            result += i.page_content
    return result


if __name__ == "__main__":
    # del_rag("weather")
    files = [
    # r"rag\9787502958572_L.pdf",    
    r"rag\气象灾害防御_示范地区作物气象灾害防御指南_气象出版社.pdf",
    r"rag\气象科普_开远市气象科普手册_气象出版社.pdf",
    r"rag\国标GBT44709-2024_旅游景区雷电灾害防御技术规范.pdf",
    r"rag\国标GBT44954-2024_山岳地区雷电灾害防御技术规范.pdf",
    r"rag\国标GBT37926-2019_美丽乡村气象防灾减灾指南.pdf",
]
    for file in files:
        add_rag_pdf(file_path=file)
    print(get_retriever("重大气象灾害之后要注意什么"))
