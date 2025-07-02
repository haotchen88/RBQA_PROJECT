import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_LANGCHAIN
from typing import List, Dict, Optional
import FAISS

@st.cache_resource
def get_embedding_function():
    """获取LangChain的嵌入函数"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_LANGCHAIN,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

@st.cache_resource
def get_vector_db():
    """返回内存版 FAISS 向量数据库"""
    embedding_function = get_embedding_function()
    # FAISS 需要先初始化为空库
    return FAISS.from_texts([], embedding_function)

def add_texts_to_db(texts: List[str], metadatas: List[Dict]):
    """添加文本和元数据到向量数据库"""
    db = get_vector_db()
    db.add_texts(texts=texts, metadatas=metadatas)

def search_db(query: str, k: int = 3) -> List:
    """在向量数据库中执行相似性搜索"""
    db = get_vector_db()
    try:
        return db.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"知识库检索失败: {str(e)}")
        return []

def delete_from_db_by_source_id(source_id: str):
    """FAISS 不支持按元数据删除，需重建索引"""
    st.warning("FAISS 不支持按元数据删除，如需此功能请用 ChromaDB。")

def clear_db():
    """清空向量数据库集合"""
    get_vector_db.clear()

def load_existing_documents() -> Optional[List[Dict]]:
    """FAISS 不支持直接加载所有文档"""
    st.warning("FAISS 不支持直接加载所有文档。")
    return None

def get_vector_count():
    """返回向量数量"""
    try:
        db = get_vector_db()
        return len(db.index_to_docstore_id)
    except Exception:
        return 0