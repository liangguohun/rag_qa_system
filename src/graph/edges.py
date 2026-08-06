"""
条件路由模块
============

LangGraph 核心概念 —— Edge（边）与 Conditional Edge（条件边）:

  普通边（add_edge）:
    固定路由，节点 A 执行完总是到节点 B。
    例如: tools → agent（工具执行完总是回到 LLM 重新思考）

  条件边（add_conditional_edges）:
    根据 State 动态决定走向，由路由函数返回目标节点名。
    例如: agent → {tools | END}（LLM 决定是需要工具还是直接回答）

本模块使用 LangGraph 预置的 tools_condition 路由函数:
  - 检查最后一条 AIMessage 是否有 tool_calls
  - 有 → 返回 "tools"（继续 ReAct 循环）
  - 无 → 返回 END（退出循环，返回结果）

ReAct 循环对照:
  Thought → Action → Observation → Thought → ... → Final Answer
      ↑                    ↑
  call_model          ToolNode

  tools_condition 负责在 Observation 之后判断：
    - 还需要更多信息？→ 回到 Thought（继续循环）
    - 已经可以回答？→ Final Answer（退出）

注意: tools_condition 是 langgraph.prebuilt 提供的标准实现，等价于以下逻辑:

    def should_continue(state):
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            return "tools"
        return END

使用预置版本的优势：框架官方维护、与 ToolNode 协作经过充分测试、支持自定义 messages_key。
"""

from langgraph.prebuilt import tools_condition

# 直接导出 tools_condition 作为模块的公共 API
# graph_builder.py 通过 from .edges import tools_condition 使用
__all__ = ["tools_condition"]
