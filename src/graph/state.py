"""
Agent 状态定义模块
==================

LangGraph 核心概念 —— State（状态）:
  State 是图中所有节点共享的数据结构。每个节点读取 State、返回部分更新的 dict，
  LangGraph 自动将返回值合并回 State。

使用 add_messages 作为 reducer（而非直接覆盖）的意义：
  默认 dict 合并是"后值覆盖前值"，而 add_messages 是"追加到列表"。
  这让 Agent 的对话历史可以自然累积：

    HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(final)

  每次 LLM 调用返回的 AIMessage 和工具执行返回的 ToolMessage 都会被追加，
  因此 LLM 始终拥有完整的上下文。

TypedDict 的作用：
  为 State 提供类型注解，LangGraph 据此推断各字段的 reducer 行为。
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Agent 共享状态。

    Attributes:
        messages: 对话消息历史
                  使用 Annotated[Sequence[BaseMessage], add_messages] 确保
                  每次节点返回 {"messages": [...]} 时是追加而非覆盖。

    消息流转示例:
        ┌─────────────────────────────────────────────────────────┐
        │ [HumanMessage]   ← 用户输入                              │
        │        │                                                │
        │ [AIMessage]      ← LLM 分析后决定调用工具 (含 tool_calls) │
        │        │                                                │
        │ [ToolMessage]    ← 工具执行返回结果                       │
        │        │                                                │
        │ [AIMessage]      ← LLM 根据工具结果生成最终回答           │
        └─────────────────────────────────────────────────────────┘

    扩展方式:
        如需添加新字段（如用户 ID、会话 ID），直接在 TypedDict 中追加即可:
          user_id: str
          session_id: str
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
