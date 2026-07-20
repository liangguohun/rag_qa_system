# 修复 Python 路径，解决找不到 src
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.rag_chain import create_rag_chain
# test_tools, 
from src.rag_chain import create_agent_with_tools, test_mcp_tools_async
from typing import Optional

# 🔥 关键：用线程运行，不让 LLM 卡住异步
import asyncio


def invoke_agent(agent, payload):
    """Invoke an agent with async support for StructuredTool-based agents."""
    if hasattr(agent, "ainvoke"):
        return asyncio.run(agent.ainvoke(payload))
    if hasattr(agent, "invoke"):
        return agent.invoke(payload)
    raise RuntimeError("当前 Agent 不支持 invoke 或 ainvoke")

# ===================== 新版 FastAPI 生命周期（无废弃警告）=====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, retriever, llm, agent, agent_tools
    # 尝试初始化 RAG 链；如果失败，只打印错误并继续运行应用
    try:
        rag_chain, retriever, llm = create_rag_chain()
        print("✅ RAG 链初始化完成")

        # 同时创建 MCP Agent，优先使用 MCP 工具
        try:
            config_file = ROOT / "config" / "mcp_servers.json"
            agent, agent_tools = await create_agent_with_tools(
                llm,
                use_mcp=True,
                mcp_config_file=str(config_file) if config_file.exists() else None
            )
            if agent is not None:
                print(f"✅ Agent 初始化完成，工具数量：{len(agent_tools)}")
            else:
                print("⚠️ Agent 初始化失败：未创建 Agent")
        except Exception as e:
            print(f"⚠️ Agent 初始化失败：{e}")
            agent, agent_tools = None, []
    except Exception as e:
        print(f"RAG 链初始化失败（继续运行）：{e}")
        import traceback
        traceback.print_exc()
        rag_chain, retriever, llm = None, None, None
        agent, agent_tools = None, []
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
llm = None
agent = None
agent_tools = []

# ===================== 请求模型 =====================
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3
    timeout: Optional[float] = 30.0

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = []

# ===================== 问答接口（新版正确调用）=====================
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        print("收到问题：", request.question)
        payload = {
            "messages": [
                {"role": "user", "content": request.question}
            ]
        }

        if agent is not None:
            answer = await asyncio.wait_for(
                asyncio.to_thread(invoke_agent, agent, payload),
                timeout=request.timeout
            )
        elif rag_chain is not None:
            answer = await asyncio.wait_for(
                asyncio.to_thread(rag_chain.invoke, request.question),
                timeout=request.timeout
            )
        else:
            raise RuntimeError("当前没有可用的 Agent 或 RAG 链")

        print("AI 返回：", answer)
        return {"answer": answer, "sources": []}
    except asyncio.TimeoutError:
        print(f"⏰ 问答超时（{request.timeout}秒）")
        raise HTTPException(
            status_code=504, 
            detail=f"问答超时，请稍后重试（超时时间：{request.timeout}秒）"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错误：{str(e)}")

# ===================== 健康检查 =====================
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


from src.rag_chain import create_agent_with_tools, load_mcp_tools
from config.mcp_config import MCP_SERVERS, load_mcp_config

# 添加MCP配置端点
@app.get("/mcp/tools")
async def get_mcp_tools():
    """获取当前MCP工具列表"""
    try:
        tools = await load_mcp_tools()
        return {
            "status": "success",
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args": tool.args_schema.schema() if tool.args_schema else {}
                }
                for tool in tools
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/mcp/reload")
async def reload_mcp_config(config_file: str = None):
    """重新加载MCP配置"""
    try:
        if config_file:
            load_mcp_config(config_file)
        else:
            # 重新加载默认配置
            from config.mcp_config import MCP_SERVERS
            # 这里重新初始化MCP客户端
            pass
        return {"status": "success", "message": "MCP配置已重新加载"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------------- 主函数 ----------------------
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.chat_models import ChatZhipuAI
from config.settings import *

# 硬编码的工具测试
''' 
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
        
'''

#  mcp 模式的工具测试
def test_mcp():
    """主函数：演示如何使用工具和RAG"""
    import asyncio
    
    # 1. 测试MCP工具（使用配置）
    print("\n🔧 测试MCP工具...")
    try:
        # 可以指定配置文件路径
        config_file = ROOT / "config" / "mcp_servers.json"
        if config_file.exists():
            asyncio.run(test_mcp_tools_async(str(config_file)))
        else:
            asyncio.run(test_mcp_tools_async())
    except Exception as e:
        print(f"MCP测试失败: {e}")

    # 2. 创建RAG链和Agent
    try:
        print("\n" + "="*50)
        print("创建 RAG 链")
        print("="*50)
        rag_chain, retriever, llm = create_rag_chain()
        
        # 创建带工具的Agent（支持MCP配置）
        print("\n" + "="*50)
        print("创建 Agent（支持MCP）")
        print("="*50)
        
        # 可以选择是否使用MCP
        use_mcp = True  # 可以改为False禁用MCP
        config_file = ROOT / "config" / "mcp_servers.json"
        
        agent, tools = asyncio.run(create_agent_with_tools(
            llm, 
            use_mcp=use_mcp,
            mcp_config_file=str(config_file) if config_file.exists() else None
        ))

        if agent is None:
            print("Agent创建失败")
            return
        
        # 测试Agent
        print("\n" + "="*50)
        print("测试 Agent 交互")
        print("="*50)
        
        test_questions = [
            "vue3 reactor 使用细节及详解",
            "帮我计算 25 * 4 + 10",
            "LangChain 这个单词文本长度？",
            "北京今天天气怎么样？",
        ]
        
        for question in test_questions:
            print(f"\n用户问题: {question}")
            try:
                response = invoke_agent(agent, {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                })
                ans = response["messages"][-1].content
                print(f"助手回答: {ans}")
                print("-" * 40)
            except Exception as e:
                print(f"Agent调用失败: {e}")
                
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 调用工具用例
    test_mcp()

    # 问答用例
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)