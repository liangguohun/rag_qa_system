"""
Agent 状态图构建器
==================

使用 LangGraph 的 StateGraph API 构建完整的 ReAct Agent 执行图。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReAct (Reasoning + Acting) 模式简介
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ReAct 是一种将"推理"和"行动"交织进行的 Agent 范式:
  1. Thought  (思考):  LLM 分析当前状态，决定下一步
  2. Action   (行动):  执行工具调用
  3. Observation (观察): 获取工具返回结果
  4. 重复 1-3，直到 LLM 认为可以给出最终答案

在 LangGraph 中，这个循环被建模为一个有向图:

        ┌─────────────────────────────────────────────┐
        │              START                          │
        │                │                            │
        │         ┌──────▼──────┐                     │
        │         │   agent     │  ← call_model 节点  │
        │         │  (LLM 推理) │    LLM 分析问题      │
        │         └──────┬──────┘   决定是否需要工具    │
        │                │                            │
        │         ┌──────▼──────┐                     │
        │         │should_      │  ← 条件边            │
        │         │continue     │   根据 tool_calls    │
        │         └──┬──────┬───┘   动态路由            │
        │            │      │                         │
        │    有工具  │      │  无工具                   │
        │   "tools" │      │  END                     │
        │    ┌───────▼──┐   │                         │
        │    │  tools    │  │  ← ToolNode 节点         │
        │    │ (执行工具) │  │    并行调用工具           │
        │    └──────┬───┘   │    返回结果               │
        │           │       │                         │
        │           └───────┘  ← 固定边: 回到 agent     │
        │                    (形成 Reasoning 循环)      │
        └─────────────────────────────────────────────┘

为什么用 LangGraph 而不是黑盒的 create_agent:
  - 显式图结构: 每一步都可追踪、可调试
  - 可扩展:  可随时添加新节点（如人工审批、安全过滤）
  - 可检查点: 支持持久化暂停/恢复（用于人工介入场景）
  - 可流式:   支持逐 token 流式输出

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import tools_condition

from .state import AgentState
from .nodes import (
    create_call_model_node,
    create_call_model_node_async,
    create_tool_node,
)
from .tools_registry import ToolRegistry


def build_agent_graph(
    llm,
    tool_registry: ToolRegistry,
    system_prompt: str = "",
    checkpointer: Optional[InMemorySaver] = None,
    async_mode: bool = True,
):
    """
    构建并编译 ReAct Agent 状态图。

    图编译流程:
      1. 从 ToolRegistry 获取工具列表 → llm.bind_tools(tools)
      2. 创建节点: agent (call_model) + tools (ToolNode)
      3. 建图: 注册节点 → 设入口 → 加条件边 → 编译
      4. 返回 CompiledStateGraph，支持 .invoke() / .ainvoke() / .stream()

    Args:
        llm:             语言模型实例（需支持 bind_tools）
        tool_registry:   工具注册中心（含已初始化的所有工具）
        system_prompt:   系统提示词，可选
        checkpointer:    检查点存储器，默认 InMemorySaver（内存）。
                         用于支持 .get_state() / 人工中断 / 恢复。
        async_mode:      是否使用异步 call_model 节点。

    Returns:
        CompiledStateGraph: LangGraph 编译后的可执行图。

    使用示例:
        from langchain_core.messages import HumanMessage

        graph = build_agent_graph(llm, registry)
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="帮我计算 25*4+10")]
        })

        # 查看完整的推理轨迹
        for msg in result["messages"]:
            print(f"[{type(msg).__name__}] {msg.content}")
    """
    # ── 1. 获取工具并绑定到 LLM ──
    tools = tool_registry.get_all_tools()

    print(f"\n[GraphBuilder] 构建 Agent 图, 工具={tool_registry.tool_count} 个")
    for t in tools:
        print(f"  - {t.name}: {t.description[:60]}...")

    # bind_tools: 将工具的 name/description/args_schema 注入 LLM 的
    # function calling 能力，使 LLM 在推理时能生成 tool_calls
    try:
        llm_with_tools = llm.bind_tools(tools)
        print("[GraphBuilder] bind_tools 成功")
    except Exception as e:
        print(f"[GraphBuilder] bind_tools 失败 ({e}), 回退到不带工具的 LLM")
        llm_with_tools = llm

    # ── 2. 创建节点 ──
    if async_mode and hasattr(llm_with_tools, "ainvoke"):
        agent_node = create_call_model_node_async(llm_with_tools)
        print("[GraphBuilder] call_model 节点: 异步模式")
    else:
        agent_node = create_call_model_node(llm_with_tools)
        print("[GraphBuilder] call_model 节点: 同步模式")

    tool_node = create_tool_node(tools) if tools else None

    # ── 3. 构建状态图 ──
    # StateGraph(AgentState): 定义图中流转的共享状态类型
    builder = StateGraph(AgentState)

    # add_node: 给图注册处理单元
    builder.add_node("agent", agent_node)
    if tool_node:
        builder.add_node("tools", tool_node)

    # set_entry_point: 标记图的起始节点
    builder.set_entry_point("agent")

    if tool_node:
        # add_conditional_edges: agent 执行后走条件路由
        # tools_condition 是 LangGraph 预置路由函数，检查最后一条 AIMessage
        # 是否有 tool_calls → 有则返回 "tools"，无则返回 END
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )
        # add_edge: tools 执行后固定回到 agent（形成循环）
        builder.add_edge("tools", "agent")
    else:
        # 无工具时: agent 直接结束
        builder.add_edge("agent", END)

    # ── 4. 编译图 ──
    # checkpointer: 持久化层, 启用后支持:
    #   - graph.get_state(config): 查看历史状态
    #   - graph.update_state(config, values): 人工修改状态（中断后注入审批结果）
    #   - 中断后从断点恢复执行
    #   注意: 启用 checkpointer 后, 每次 .ainvoke() 必须传入
    #   config={"configurable": {"thread_id": "xxx"}} 以区分会话.
    if checkpointer is None:
        checkpointer = InMemorySaver()

    compiled_graph = builder.compile(checkpointer=checkpointer)
    print("[GraphBuilder] 图编译完成")

    return compiled_graph
