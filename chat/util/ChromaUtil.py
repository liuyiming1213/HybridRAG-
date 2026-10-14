# 项目根路径
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

path_base = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 向量化模型路径
embedding_model_name = os.path.abspath(os.path.join(path_base, 'models', 'embedding_model'))
# 向量数据库存储路径
vector_db_path = os.path.abspath(os.path.join(path_base, 'vector_db'))
# 向量数据库集合的名字
vector_db_name = 'worldcup_kb'


# 获取向量数据库连接对象（用于知识库检索）
def get_kb_conn():
    # 向量化模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={
            "local_files_only": True,
            "device": "cpu"
        },
    )
    # 获取向量数据库检索对象
    return Chroma(
        persist_directory=vector_db_path,
        collection_name=vector_db_name,
        embedding_function=embedding_model,
    )