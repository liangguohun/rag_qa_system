# 修复 Python 路径，解决找不到 src
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag_chain import create_rag_chain
from typing import Optional

# 🔥 关键：用线程运行，不让 LLM 卡住异步
import asyncio

# ===================== 新版 FastAPI 生命周期（无废弃警告）=====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, retriever
    # 这里返回 2 个值！！！
    rag_chain, retriever = create_rag_chain()
    print("✅ RAG 链初始化完成")
    yield
    print("🛑 服务关闭")

app = FastAPI(
    title="RAG知识库问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 全局变量
rag_chain = None
retriever = None

# ===================== 请求模型 =====================
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = []

# ===================== 问答接口（新版正确调用）=====================
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        print("收到问题：", request.question)
        # 新版调用：.invoke()
        # answer = rag_chain.invoke(request.question)

        answer = await asyncio.to_thread(
            rag_chain.invoke,
            request.question
        )

        print("AI 返回：", answer)
        return {"answer": answer, "sources": []}
    
        # # 获取来源文档
        # source_docs = retriever.invoke(request.question)
        # sources = [doc.metadata.get("source", "unknown") for doc in source_docs]

        # return QueryResponse(
        #     answer=answer,
        #     sources=sources
        # )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错误：{str(e)}")

# ===================== 健康检查 =====================
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)