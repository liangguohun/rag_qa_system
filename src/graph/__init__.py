"""
LangGraph Agent 模块
====================
基于 LangGraph 状态图（StateGraph）实现可解释、可扩展的 ReAct Agent。

核心组件：
  AgentState        - Agent 状态定义（含消息列表的共享状态）
  ToolRegistry      - 工具注册中心（管理 MCP / RAG / 本地工具的生命周期）
  build_agent_graph - 构建并编译 ReAct 循环状态图

快速使用：
  >>> from src.graph import ToolRegistry, build_agent_graph
  >>> from langchain_core.messages import HumanMessage
  >>>
  >>> registry = ToolRegistry()
  >>> await registry.initialize(use_mcp=True, retriever=my_retriever)
  >>>
  >>> graph = build_agent_graph(llm, registry, system_prompt="你是一个助手")
  >>>
  >>> result = await graph.ainvoke({
  ...     "messages": [HumanMessage(content="你好")]
  ... })
  >>> print(result["messages"][-1].content)
"""

from .state import AgentState
from .tools_registry import ToolRegistry
from .graph_builder import build_agent_graph

__all__ = [
    "AgentState",
    "ToolRegistry",
    "build_agent_graph",
]
