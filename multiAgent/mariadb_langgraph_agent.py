"""
mariadb_langgraph_agent.py
基于 MariaDB MCP Server 的 LangGraph 数据库 Agent，按 config/settings.py 的开关接入：

    MARIADB_MCP_TRANSPORT = "http"   → 自动拉起本地 vendor/mariadb-mcp 的 HTTP Service，
                                       MCP 用 streamable-http 连接
    MARIADB_MCP_TRANSPORT = "stdio"  → MCP 用 stdio 拉起内嵌子进程（npx / uv）

服务启停逻辑复用 src/mariadb_mcp_service（与 main.py 保持一致）。

运行示例：
    python multiAgent/mariadb_langgraph_agent.py
    python multiAgent/mariadb_langgraph_agent.py --question "查询用户表前5行"
"""
import argparse
import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以导入 config / src 包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import (
    MARIADB_DB_CONFIG,
    MARIADB_MCP_STDIO_COMMAND,
    MARIADB_MCP_STDIO_ARGS,
    MARIADB_MCP_HTTP_HEADERS,
    MARIADB_LLM_MODEL,
    MARIADB_LLM_BASE_URL,
    MARIADB_LLM_API_KEY,
)
from src.mariadb_mcp_service import (
    ensure_http_server,
    get_mcp_url,
    get_transport,
    stop_mariadb_mcp_service,
)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


# ==================== LangGraph ====================

async def build_graph(transport: str):
    """按传输模式构建 LangGraph（MCP 客户端），返回 (graph, mcp_client)"""
    if transport == "stdio":
        server_config = {
            "mariadb": {
                "transport": "stdio",
                "command": MARIADB_MCP_STDIO_COMMAND,
                "args": MARIADB_MCP_STDIO_ARGS,
                # 把数据库连接环境变量透传给 mcp 子进程
                "env": MARIADB_DB_CONFIG,
            }
        }
        print("[mariadb-langgraph] stdio 模式，拉起命令: "
              + f"{MARIADB_MCP_STDIO_COMMAND} {' '.join(MARIADB_MCP_STDIO_ARGS)}")
    else:  # http / streamable-http
        server_config = {
            "mariadb": {
                "transport": "streamable_http",
                "url": get_mcp_url(),
                "headers": MARIADB_MCP_HTTP_HEADERS,
            }
        }
        print(f"[mariadb-langgraph] streamable-http 模式，连接: {get_mcp_url()}")

    # 获取 mcp 暴露的数据库工具
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()
    print("[mariadb-langgraph] 已加载 MCP 工具 "
          + f"{len(tools)} 个: {[t.name for t in tools]}")

    llm = ChatOpenAI(
        model=MARIADB_LLM_MODEL,
        api_key=MARIADB_LLM_API_KEY,
        base_url=MARIADB_LLM_BASE_URL,
    )

    def call_model(state: MessagesState):
        resp = llm.bind_tools(tools).invoke(state["messages"])
        return {"messages": [resp]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges("call_model", tools_condition)
    graph_builder.add_edge("tools", "call_model")

    graph = graph_builder.compile()
    return graph, client


# ==================== 入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MariaDB MCP + LangGraph 数据库 Agent（传输模式由 config/settings.py 的 "
                    "MARIADB_MCP_TRANSPORT 决定：http / stdio）"
    )
    _ = parser.add_argument("--question", help="自定义查询问题（默认查询 DB_NAME 库表）")
    return parser.parse_args()


async def main():
    transport = get_transport()
    args = parse_args()
    mcp_client = None
    http_proc = None
    try:
        # http 模式：自动拉起本地 MariaDB MCP HTTP Service 后访问
        if transport == "http":
            http_proc = await ensure_http_server()

        graph, mcp_client = await build_graph(transport)

        question = args.question or (
            f"列出 {MARIADB_DB_CONFIG['DB_NAME']} 库所有表，查看第一张表的结构"
        )
        result = await graph.ainvoke({
            "messages": [
                SystemMessage(content="你是数据库助手，使用mariadb工具查询数据库，只做查询，不要写修改数据SQL。"),
                {"role": "human", "content": question},
            ]
        })
        for msg in result["messages"]:
            msg.pretty_print()
    finally:
        if mcp_client is not None:
            await mcp_client.close()
        await stop_mariadb_mcp_service(http_proc)


if __name__ == "__main__":
    asyncio.run(main())
