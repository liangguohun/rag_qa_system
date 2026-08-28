"""
节点（Node）模块
================

LangGraph 核心概念 —— Node（节点）:
  节点是图中的执行单元，每个节点是一个纯函数:
      f(State) -> dict     （返回需要更新的 State 子集）

  LangGraph 不关心节点内部的实现细节，只关心 State 进、dict 出。

本模块包含两个核心节点：

1. call_model（Agent 节点）
   - 将当前 messages 发送给已绑定工具 schema 的 LLM
   - LLM 返回 AIMessage，可能包含:
     * tool_calls: 需要调用工具（继续循环）
     * content:    直接文本回答（循环结束）
   - 此节点实现了 ReAct 框架中的 "Reasoning" 阶段

2. ToolNode（工具节点）
   - LangGraph 预置节点，自动完成:
     ① 从最后一条 AIMessage 提取 tool_calls
     ② 按名称匹配工具并并行执行
     ③ 返回 ToolMessage 列表
   - 此节点实现了 ReAct 框架中的 "Acting" 阶段

ReAct 循环对照:
  Thought → Action → Observation → Thought → ... → Final Answer
     ↑                    ↑
  call_model          ToolNode
"""

import asyncio
import time

from openai import APIError, RateLimitError

from langgraph.prebuilt import ToolNode

# ── LLM 限流/服务端错误退避重试参数 ──
MAX_LLM_RETRIES = 3          # 最多尝试次数（含首次）
LLM_RETRY_BASE_DELAY = 2.0   # 首次重试等待秒数（指数增长）
LLM_RETRY_MAX_DELAY = 30.0   # 单次最大等待秒数


def _should_retry_llm(exc) -> bool:
    """是否值得退避重试：429 限流 / 5xx 服务端错误 / 网络连接错误"""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        if code is None:  # 连接层错误（APIConnectionError 等）
            return True
        return 500 <= code < 600
    return False


# ============================================================
# call_model 节点工厂
# ============================================================

def create_call_model_node(llm_with_tools):
    """
    创建同步版 "Agent 思考" 节点。

    执行逻辑:
        1. 从 State 中取出 messages（完整对话历史）
        2. 调用 llm_with_tools.invoke(messages)
        3. LLM 根据绑定的工具 schema 决定:
           - 生成 tool_calls → Agent 将进入 tools 节点
           - 生成 text      → Agent 将结束循环
        4. 返回 {"messages": [AIMessage]}，由 add_messages reducer 追加

    Args:
        llm_with_tools: 已通过 bind_tools() 绑定工具 schema 的 LLM 实例。
                        bind_tools 将工具的 name/description/args_schema 注入 LLM 的
                        function calling 能力，使 LLM 在需要时自动生成 tool_calls。

    Returns:
        callable: 节点函数 f(state, config=None) -> dict
    """
    def call_model(state, config=None):
        messages = state.get("messages", [])
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except APIError as e:
                if not _should_retry_llm(e) or attempt >= MAX_LLM_RETRIES:
                    raise
                delay = min(LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)), LLM_RETRY_MAX_DELAY)
                print(f"\033[33m[LLM] 调用受限（{getattr(e, 'status_code', '连接错误')}），"
                      f"{delay:.0f}s 后重试 ({attempt}/{MAX_LLM_RETRIES})\033[0m")
                time.sleep(delay)

    return call_model


def create_call_model_node_async(llm_with_tools):
    """
    创建异步版 "Agent 思考" 节点。

    与同步版逻辑完全相同，使用 ainvoke 替代 invoke。
    当 LLM 是异步模型（如 ChatOpenAI with async client）时，
    使用此节点可避免阻塞事件循环。

    Args:
        llm_with_tools: 已 bind_tools 的异步 LLM 实例。

    Returns:
        async callable: 节点函数 async(state, config=None) -> dict
    """
    async def call_model(state, config=None):
        messages = state.get("messages", [])
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                response = await llm_with_tools.ainvoke(messages)
                return {"messages": [response]}
            except APIError as e:
                if not _should_retry_llm(e) or attempt >= MAX_LLM_RETRIES:
                    raise
                delay = min(LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)), LLM_RETRY_MAX_DELAY)
                print(f"\033[33m[LLM] 调用受限（{getattr(e, 'status_code', '连接错误')}），"
                      f"{delay:.0f}s 后重试 ({attempt}/{MAX_LLM_RETRIES})\033[0m")
                await asyncio.sleep(delay)

    return call_model


# ============================================================
# ToolNode 节点工厂
# ============================================================

def create_tool_node(tools: list):
    """
    创建 "工具执行" 节点。

    ToolNode 是 LangGraph 内置的高层封装，自动处理:
        ┌─────────────────────────────────────────────────┐
        │ 1. 找到 state.messages[-1] 中的 tool_calls       │
        │ 2. 按 tool_calls[].name 匹配 tools 列表中的工具   │
        │ 3. 以 tool_calls[].args 为参数并行调用所有工具     │
        │ 4. 将每个工具返回值包装为 ToolMessage             │
        │ 5. 返回 {"messages": [ToolMessage, ...]}         │
        └─────────────────────────────────────────────────┘

    工具调用链示例:
        输入 AIMessage:
          tool_calls = [
            {"name": "calculate", "args": {"expression": "1+1"}, "id": "call_001"},
            {"name": "weather_check", "args": {"city": "北京"}, "id": "call_002"}
          ]

        ToolNode 执行:
          calculate.invoke({"expression": "1+1"})   → "2"
          weather_check.invoke({"city": "北京"})      → "北京: 晴天，25°C..."

        输出 ToolMessages:
          [ToolMessage(content="2", tool_call_id="call_001"),
           ToolMessage(content="北京: 晴天...", tool_call_id="call_002")]

    Args:
        tools: 工具列表，每个工具必须有 .name 和 .invoke/.ainvoke 方法

    Returns:
        ToolNode: 可添加到 StateGraph 的节点
    """
    return ToolNode(tools)
