## 运行

```
激活环境
conda activate langchain_env 
cd F:\aiDemo\rag_qa_system

运行应用
python ./main.py
```

## 智谱 模型配置

```
1、安装依赖
pip install -U zhipuai

2、settings.py 

# ========== 智谱 AI 免费模型配置 ==========
USE_LOCAL_LLM = False  # 关闭本地模型
USE_ZHIPU = True       # 开启智谱免费模型

# 智谱 AI 密钥（去官网免费拿）
ZHIPU_API_KEY = "你的智谱API_KEY"

# 免费模型名称（固定不要改）
ZHIPU_MODEL_NAME = "glm-4-flash"

# 嵌入模型（免费、轻量、中文强）
ZHIPU_EMBEDDING_MODEL = "embedding-2"

3、rag_chain.py

from langchain_community.chat_models import ChatZhipuAI
from zhipuai import ZhipuAI  # 官方SDK

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
# ================== 智谱免费版（推荐） ==================
if USE_LOCAL_LLM:
    embeddings = OllamaEmbeddings(model=LOCAL_EMBEDDING_MODEL)
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



```