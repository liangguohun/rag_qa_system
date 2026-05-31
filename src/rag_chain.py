# 修复：文本分割器新路径
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ✅ 新版 Ollama 包（无废弃警告）
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 新版 RAG 链核心导入
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from config.settings import *
import os


def create_rag_chain():
    """创建 RAG 链（LangChain v1.0+ 无警告版）"""
    
    # 1. 加载文档
    from src.loaders import load_documents_from_directory
    documents = load_documents_from_directory(RAW_DATA_DIR)
    
    # 2. 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. 向量化（✅ 新版 Ollama 无警告）
    if USE_LOCAL_LLM:
        # ✅ 新版 Ollama 嵌入模型（无废弃警告）
        embeddings = OllamaEmbeddings(model=LOCAL_EMBEDDING_MODEL)
        # ✅ 新版 Ollama LLM（无废弃警告）
        llm = OllamaLLM(model=LOCAL_LLM_MODEL, temperature=TEMPERATURE)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=TEMPERATURE)
    
    # 4. 创建向量库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # ===================== 新版 RAG 链 =====================
    prompt = PromptTemplate.from_template("""
使用以下上下文来回答最后的问题。
如果你不知道答案，就直接说不知道，不要编造答案。

{context}

问题：{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    # ======================================================
    
    return rag_chain, retriever