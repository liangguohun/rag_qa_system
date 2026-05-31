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

from langchain_community.chat_models import ChatZhipuAI
from zhipuai import ZhipuAI  # 官方SDK

from config.settings import *
import os

# ---------------------- 修复版的智谱 Embedding ----------------------
class SafeZhipuEmbeddings:
    def __init__(self, api_key: str, model: str = "embedding-2"):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
    
    def _clean_text(self, text: str) -> str:
        """清理和截断文本"""
        if not text or not isinstance(text, str):
            return ""
        # 去除首尾空白，限制长度（embedding-2 最大 512 tokens，约 1000-1500 字符）
        cleaned = text.strip()[:1500]
        return cleaned
    
    def embed_documents(self, texts):
        """批量向量化文档"""
        # 过滤空文本并清理
        safe_texts = []
        for t in texts:
            cleaned = self._clean_text(t)
            if cleaned:  # 只添加非空文本
                safe_texts.append(cleaned)
        
        if not safe_texts:
            print("警告：没有有效的文本需要向量化")
            return []
        
        try:
            # 批量处理，每次最多 32 条（API 限制）
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(safe_texts), batch_size):
                batch = safe_texts[i:i+batch_size]
                print(f"正在向量化第 {i//batch_size + 1} 批，共 {len(batch)} 条文本")
                
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # 提取 embeddings，确保顺序正确
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
            
            print(f"成功向量化 {len(all_embeddings)} 条文本")
            return all_embeddings
            
        except Exception as e:
            print(f"Embedding调用失败：{e}")
            # 打印更多错误信息用于调试
            if hasattr(e, 'response'):
                print(f"响应内容：{e.response.text}")
            raise
    
    def embed_query(self, text):
        """向量化单个查询"""
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("查询文本为空")
        
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=[cleaned]  # 注意：必须是列表格式
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"查询向量化失败：{e}")
            raise
# ---------------------------------------------------------------------------

def create_rag_chain():
    """创建 RAG 链（LangChain v1.0+ 无警告版）"""
    
    # 1. 加载文档
    from src.loaders import load_documents_from_directory
    documents = load_documents_from_directory(RAW_DATA_DIR)
    
    # 检查文档是否为空
    if not documents:
        raise ValueError(f"在 {RAW_DATA_DIR} 中没有找到任何文档")
    
    print(f"加载了 {len(documents)} 个文档")
    
    # 2. 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise ValueError("文档分块后为空")
    
    print(f"文档分块为 {len(chunks)} 个块")
    
    # 3. 向量化（✅ 新版 Ollama 无警告）
    if USE_LOCAL_LLM:
        # ✅ 新版 Ollama 嵌入模型（无废弃警告）
        embeddings = OllamaEmbeddings(model=LOCAL_EMBEDDING_MODEL)
        # ✅ 新版 Ollama LLM（无废弃警告）
        llm = OllamaLLM(model=LOCAL_LLM_MODEL, temperature=TEMPERATURE)
    else:
        # ✅ 用修复后的智谱 Embedding
        embeddings = SafeZhipuEmbeddings(
            api_key=ZHIPU_API_KEY, 
            model=ZHIPU_EMBEDDING_MODEL
        )
        # ✅ LLM用官方ChatZhipuAI
        llm = ChatZhipuAI(
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL_NAME,
            temperature=TEMPERATURE
        )
    
    # 4. 创建向量库
    print("正在创建向量数据库...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )
    
    print("向量数据库创建完成")
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # ===================== 新版 RAG 链 =====================
    prompt = PromptTemplate.from_template("""
使用以下上下文来回答最后的问题。
如果你不知道答案，就直接说不知道，不要编造答案。

上下文：
{context}

问题：{question}

回答：
""")

    def format_docs(docs):
        if not docs:
            return "没有找到相关上下文"
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

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