# 修复 Python 路径，解决找不到 src
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag_chain import create_rag_chain
from src.rag_chain import test_tools, create_agent_with_tools
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

# ---------------------- 主函数 ----------------------
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.chat_models import ChatZhipuAI
from config.settings import *
def main():
    """主函数：演示如何使用工具和 RAG"""
    
    # 测试简单的工具
    test_tools()
    
    # 创建 RAG 链（如果需要）
    try:
        print("\n" + "="*50)
        print("创建 RAG 链")
        print("="*50)
        rag_chain, retriever, llm = create_rag_chain()
        
        # 创建带工具的 Agent
        print("\n" + "="*50)
        print("创建 Agent（带工具）")
        print("="*50)
        # 创建 Agent
        agent, tools = create_agent_with_tools(llm)
        
        # 测试 Agent 的交互能力
        print("\n" + "="*50)
        print("测试 Agent 交互")
        print("="*50)
        
        test_questions = [
            "现在几点了？",
            "帮我计算 25 * 4 + 10",
            "LangChain 这个单词文本长度？",
            "北京今天天气怎么样？",
            "上海今天天气怎么样？",
        ]
        # 提问，Agent 会自动选择合适的工具
        for question in test_questions:
            print(f"\n用户问题: {question}")
            # response = agent.invoke({"input": question})
            # print(f"助手回答: {response['output']}")
            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": question}
                ]
            })
            ans = response["messages"][-1].content
            print(ans)
            print("-" * 40)
            
    except Exception as e:
        print(f"创建 RAG 链失败: {e}")
        print("将只演示工具功能")
        
        # 如果 RAG 失败，只创建一个简单的 LLM Agent
        if USE_LOCAL_LLM:
            llm = OllamaLLM(model=LOCAL_LLM_MODEL, temperature=TEMPERATURE)
        else:
            # ✅ LLM用官方ChatZhipuAI
            llm = ChatZhipuAI(
                api_key=ZHIPU_API_KEY,
                model=ZHIPU_MODEL_NAME,
                temperature=TEMPERATURE
            )
        
        agent, tools = create_agent_with_tools(llm)
        
        # 简单测试
        print("\n简单测试 Agent:")
        # response = agent.invoke({"input": "现在几点了？"})
        # print(f"回答: {response['output']}")
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": "现在几点了？"}
            ]
        })
        ans = response["messages"][-1].content
        print(ans)
        


if __name__ == "__main__":
    # 调用工具用例
    main()

    # 问答用例
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)