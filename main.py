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
# MariaDB MCP Service 启停管理（http 伴随启动 / stdio 延后启动）
from src.mariadb_mcp_service import start_mariadb_mcp_service, stop_mariadb_mcp_service

# 🔥 关键：用线程运行，不让 LLM 卡住异步
import asyncio
import uuid


def _agent_config(thread_id: str = None) -> dict:
    """生成 Agent 调用所需的 config（含 thread_id，供 checkpointer 使用）。"""
    return {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}


async def invoke_agent(agent, payload, thread_id: str = None):
    """异步调用 Agent（用于已在事件循环中的场景）。

    使用 asyncio.to_thread 在独立线程中执行 sync invoke()，
    避免 RedisSaver 的 async 方法未实现问题（aget_tuple 等抛 NotImplementedError）。

    Args:
        agent:    CompiledStateGraph 实例
        payload:  {"messages": [...]}
        thread_id: 会话标识（checkpointer 需要）。不传则自动生成 UUID。
    """
    config = _agent_config(thread_id)
    if hasattr(agent, "invoke"):
        return await asyncio.to_thread(agent.invoke, payload, config=config)
    if hasattr(agent, "ainvoke"):
        return await agent.ainvoke(payload, config=config)
    raise RuntimeError("当前 Agent 不支持 invoke 或 ainvoke")


def extract_tool_answer(response: dict) -> str:
    """从 Agent 响应中提取工具真实返回值作为最终答案。
    
    Agent 流程: HumanMsg → AI(tool_call) → ToolMsg(真实结果) → AI(最终回答)
    最后一个 AI 消息可能被 LLM 篡改，这里优先取 ToolMessage 中的真实数据。
    """
    messages = response.get("messages", [])
    
    # 收集所有 ToolMessage 中的文本
    tool_results = []
    for msg in messages:
        type_name = type(msg).__name__
        if type_name == "ToolMessage":
            content = getattr(msg, "content", "")
            # content 可能是字符串或列表
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        tool_results.append(item["text"])
                    elif isinstance(item, str):
                        tool_results.append(item)
            elif isinstance(content, str) and content:
                tool_results.append(content)
    
    if tool_results:
        return "\n".join(tool_results)
    
    # 没有工具结果，回退到 LLM 的最后回答
    last_msg = messages[-1] if messages else None
    if last_msg and hasattr(last_msg, "content"):
        return last_msg.content
    return "无法获取回答"


def invoke_agent_sync(agent, payload, thread_id: str = None):
    """同步调用 Agent（用于线程 / 非事件循环场景）。

    直接使用 sync invoke()，避免 RedisSaver 的 async 方法未实现问题。
    """
    config = _agent_config(thread_id)
    if hasattr(agent, "invoke"):
        return agent.invoke(payload, config=config)
    if hasattr(agent, "ainvoke"):
        return asyncio.run(agent.ainvoke(payload, config=config))
    raise RuntimeError("当前 Agent 不支持 invoke 或 ainvoke")

# ===================== 新版 FastAPI 生命周期（无废弃警告）=====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, retriever, llm, agent, agent_tools, mcp_client, mariadb_mcp_proc
    # ── MariaDB MCP Service：按 settings.py 开关管理 ──
    #   http  → 伴随启动：拉起本地 vendor/mariadb-mcp 的 HTTP Service
    #   stdio → 延后启动：不启动外部服务，由 Agent 调用时内嵌拉起
    try:
        mariadb_mcp_proc = await start_mariadb_mcp_service()
    except Exception as e:
        print(f"⚠️ MariaDB MCP Service 启动失败（不影响主服务）：{e}")
        mariadb_mcp_proc = None

    # 尝试初始化 RAG 链；如果失败，只打印错误并继续运行应用
    try:
        rag_chain, retriever, llm = create_rag_chain()
        if rag_chain is not None:
            print("✅ RAG 链初始化完成")
        else:
            print("ℹ️  RAG 检索已关闭（ENABLE_RAG_RETRIEVAL=False），仅使用纯 LLM 对话")

        # 同时创建 MCP Agent，优先使用 MCP 工具
        try:
            config_file = ROOT / "config" / "mcp_servers.json"
            agent, agent_tools, mcp_client = await create_agent_with_tools(
                llm,
                use_mcp=True,
                mcp_config_file=str(config_file) if config_file.exists() else None,
                retriever=retriever,
            )
            if agent is not None:
                print(f"✅ Agent 初始化完成，工具数量：{len(agent_tools)}")
            else:
                print("⚠️ Agent 初始化失败：未创建 Agent")
        except Exception as e:
            print(f"⚠️ Agent 初始化失败：{e}")
            agent, agent_tools, mcp_client = None, [], None

        # ── 启动完成后执行 MCP 工具测试 ──
        if agent is not None:
            print("\n" + "=" * 50)
            print("测试 Agent MCP 工具调用")
            print("=" * 50)

            # ── 前置诊断：直接调用 MCP 工具验证连接 ──
            # 各工具的连通性测试参数（btc 库含 market 表，为真实对象）
            _tool_test_args = {
                "get_current_time": {"format_type": "full"},
                "calculate": {"expression": "1+1"},
                "get_word_length": {"word": "test"},
                "weather_check": {"city": "北京"},
                "list_databases": {},
                "list_tables": {"database_name": "btc"},
                "get_table_schema": {"database_name": "btc", "table_name": "market"},
                "get_table_schema_with_relations": {"database_name": "btc", "table_name": "market"},
                "execute_sql": {"database_name": "btc", "sql_query": "SHOW TABLES FROM btc"},
                "create_database": {"database_name": "mcp_test_conn"},
            }
            print("\n[诊断] 直接测试 MCP 工具连通性（不走 Agent）:")
            for t in agent_tools:
                _args = _tool_test_args.get(t.name)
                if _args is None:
                    print(f"  ⏭️ {t.name} 无测试参数，跳过")
                    continue
                try:
                    print(f"  测试工具 {t.name}...")
                    r = await t.ainvoke(_args) if hasattr(t, 'ainvoke') else t.invoke(_args)
                    print(f"    ✅ {t.name} 返回: {r}")
                    # create_database 有副作用：测试成功后立即清理测试库
                    if t.name == "create_database":
                        for _t2 in agent_tools:
                            if _t2.name == "execute_sql":
                                await _t2.ainvoke({
                                    "database_name": _args["database_name"],
                                    "sql_query": f"DROP DATABASE IF EXISTS `{_args['database_name']}`",
                                })
                                print(f"    🧹 已清理测试库 {_args['database_name']}")
                                break
                except Exception as e:
                    _err = str(e)
                    if "read-only" in _err.lower() or "forbidden" in _err.lower():
                        print(f"    ℹ️ {t.name} 服务端只读模式拒绝（非参数错误）: {_err[:100]}")
                    else:
                        print(f"    ❌ {t.name} 直接调用失败: {e}")

            test_questions = [
                ("get_current_time", "请用 get_current_time 工具获取当前完整时间"),
                # ("calculate", "请用 calculate 工具计算 25 * 4 + 10"),
                # ("get_word_length", "请用 get_word_length 工具统计 'LangChain' 这个词的字符长度"),
                # ("weather_check", "请用 weather_check 工具查询北京的天气"),
            ]
            for tool_name, question in test_questions:
                print(f"\n[MCP:{tool_name}] 用户问题: {question}")
                try:
                    response = await invoke_agent(agent, {
                        "messages": [
                            {"role": "user", "content": question}
                        ]
                    })
                    # ── 诊断：打印完整消息链 ──
                    print(f"[诊断] Agent 返回 {len(response['messages'])} 条消息:")
                    for i, msg in enumerate(response["messages"]):
                        msg_type = type(msg).__name__
                        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                        content_preview = str(getattr(msg, 'content', ''))[:200]
                        print(f"  [{i}] type={msg_type}, content={content_preview}")
                        if has_tool_calls:
                            print(f"       tool_calls={msg.tool_calls}")
                    # ── 使用工具真实返回值，不信任 LLM 总结 ──
                    ans = extract_tool_answer(response)
                    llm_answer = response["messages"][-1].content
                    if ans != llm_answer:
                        print(f"[修正] LLM回答被篡改: {llm_answer}")
                    print(f"最终回答: {ans}")
                    print("-" * 40)
                except Exception as e:
                    # 控制台只输出带颜色的简洁错误，不打印冗长 traceback
                    print(f"\033[31m[Agent] 调用失败: {e}\033[0m")
    except Exception as e:
        print(f"RAG 链初始化失败（继续运行）：{e}")
        import traceback
        traceback.print_exc()
        rag_chain, retriever, llm = None, None, None
        agent, agent_tools, mcp_client = None, [], None
    yield
    # 关闭伴随启动的本地 MariaDB MCP HTTP Service
    try:
        await stop_mariadb_mcp_service(mariadb_mcp_proc)
    except Exception as e:
        print(f"⚠️ MariaDB MCP Service 停止异常：{e}")
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
mcp_client = None
mariadb_mcp_proc = None

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
            response = await asyncio.wait_for(
                asyncio.to_thread(invoke_agent_sync, agent, payload),
                timeout=request.timeout
            )
            answer = extract_tool_answer(response) if isinstance(response, dict) else response
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
        tools, _ = await load_mcp_tools()
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
        
        agent, tools, _mcp_client = asyncio.run(create_agent_with_tools(
            llm, 
            use_mcp=use_mcp,
            mcp_config_file=str(config_file) if config_file.exists() else None,
            retriever=retriever,
        ))

        if agent is None:
            print("Agent创建失败")
            return

        print(f"✅ Agent 创建完成，工具数量：{len(tools)}")

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