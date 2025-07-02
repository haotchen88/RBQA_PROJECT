import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_LANGCHAIN
from typing import List, Dict, Optional

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
    """返回内存版 FAISS 向量数据库（延迟初始化）"""
    # 不要用空文本初始化
    return None

def add_texts_to_db(texts: List[str], metadatas: List[Dict]):
    """添加文本和元数据到向量数据库"""
    db = st.session_state.get("vector_db")
    embedding_function = get_embedding_function()
    if db is None:
        # 首次添加，初始化
        db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)
        st.session_state.vector_db = db
    else:
        db.add_texts(texts=texts, metadatas=metadatas)

def search_db(query: str, k: int = 3) -> List:
    """在向量数据库中执行相似性搜索"""
    db = st.session_state.get("vector_db")
    if db is None:
        return []
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
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]

def load_existing_documents() -> Optional[List[Dict]]:
    """FAISS 不支持直接加载所有文档"""
    st.warning("FAISS 不支持直接加载所有文档。")
    return None

def get_document_count() -> int:
    """获取向量数据库中的文档数量"""
    db = st.session_state.get("vector_db")
    if db is None:
        return 0
    try:
        return len(db.index_to_docstore_id)
    except Exception:
        return 0

def get_vector_count() -> int:
    """返回当前向量库中的向量数量"""
    db = st.session_state.get("vector_db")
    if db is None:
        return 0
    try:
        return len(db.index_to_docstore_id)
    except Exception:
        return 0