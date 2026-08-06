"""
工具注册中心（Tool Registry）模块
================================

职责：
  1. 管理 Agent 可用的所有工具（MCP 远程 / RAG 检索 / 本地内置）
  2. 处理 MCP 工具的生命周期（连接 → 加载 → 保持存活 → 释放）
  3. 工具去重：MCP 工具优先于同名本地工具
  4. 向 Agent 暴露统一的 get_all_tools() 接口

三层工具栈:

  优先级 1 - MCP 工具（最高优先级）
    ↳ 通过 MultiServerMCPClient 从外部 MCP 服务器动态加载
    ↳ 支持 stdio / SSE 两种传输方式
    ↳ 工具 schema 由 MCP 服务端定义，客户端无需硬编码
    ↳ 举例: get_current_time, calculate, weather_check

  优先级 2 - RAG 工具
    ↳ 封装 retriever.invoke(query)，将自然语言转为向量检索
    ↳ 返回 top-k 相关文档片段
    ↳ 举例: search_knowledge_base

  优先级 3 - 本地工具（兜底）
    ↳ 硬编码的系统内置工具，保证在最坏情况下也有基本能力
    ↳ 当 MCP 服务器不可达时自动激活
    ↳ 举例: get_current_time, calculate, get_word_length, weather_check

工具去重策略:
  如果 MCP 工具和本地工具同名（如 get_current_time），则:
  - 仅保留 MCP 版本（远程工具通常功能更丰富或在服务端更新）
  - 本地同名工具被自动过滤
"""

import asyncio
from typing import Optional, List

from langchain_core.tools import tool, StructuredTool

# MCP 配置导入（项目根目录已在 sys.path 中）
from config.mcp_config import (
    load_mcp_config,
    get_enabled_servers,
    MCP_SERVERS,
    ENABLED_SERVERS,
    ENABLED_TOOLS,
)


# ============================================================
# 本地工具定义
# ============================================================

@tool
def get_current_time(format_type: str = "full") -> str:
    """获取当前日期和时间。format_type: 'full'(完整)/'date'(日期)/'time'(时间)"""
    import datetime
    now = datetime.datetime.now()
    if format_type == "date":
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """安全计算数学表达式。expression: 如 '25*4+10'"""
    import math
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names["__builtins__"] = {}
        code = compile(expression, "<calc>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return f"错误: 不允许使用 '{name}'"
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_word_length(word: str) -> str:
    """统计文本的字符长度。word: 要统计的文本"""
    return str(len(word))


@tool
def weather_check(city: str = "北京") -> str:
    """查询指定城市的天气（模拟数据）。city: 城市名，如 '北京'、'上海'"""
    weather_data = {
        "北京": "晴天，25°C，湿度45%",
        "上海": "多云，28°C，湿度65%",
        "广州": "雷阵雨，30°C，湿度80%",
        "深圳": "阴天，27°C，湿度70%",
    }
    return weather_data.get(city, f"{city}: 暂无天气数据")


# 本地工具集合（兜底列表）
LOCAL_TOOLS = [get_current_time, calculate, get_word_length, weather_check]


# ============================================================
# RAG 检索工具构造
# ============================================================

def create_rag_tool(retriever) -> Optional[StructuredTool]:
    """
    将 RAG 检索器封装为 LangChain 工具。

    原理:
      retriever.invoke(query) 内部:
        1. 将 query 向量化（embedding）
        2. 在向量库中做 ANN（近似最近邻）搜索
        3. 返回 top-k 最相似的文档片段

    封装为工具后，LLM 可以在需要时自动调用:
      → 判断问题涉及专业知识 → 生成 tool_call → search_knowledge_base → 返回文档片段

    Args:
        retriever: LangChain Retriever 实例（如 Chroma.as_retriever()）

    Returns:
        StructuredTool | None: 如果 retriever 为 None 则返回 None
    """
    if retriever is None:
        return None

    def search_kb(query: str) -> str:
        """在 RAG 知识库中检索相关文档内容"""
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中未找到相关信息"
        return "\n\n---\n\n".join(
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        )

    return StructuredTool.from_function(
        func=search_kb,
        name="search_knowledge_base",
        description="在 RAG 知识库中检索相关文档内容。当问题涉及专业概念、技术原理、文档内容时优先使用此工具。参数 query: 检索查询词。",
    )


# ============================================================
# MCP 工具动态加载
# ============================================================

async def load_mcp_tools(config_file: str = None):
    """
    从 MCP 服务器动态加载工具列表。

    加载流程:
      1. 读取 MCP 配置（mcp_servers.json 或默认 config/mcp_config.py 中的配置）
      2. 过滤出启用的服务器
      3. 为每个服务器创建 MultiServerMCPClient 会话
      4. 调用 client.get_tools() 获取远程工具 schema
      5. 根据 ENABLED_TOOLS 字典过滤（工具级别的启用/禁用）

    关键设计:
      client 对象必须由调用方保持存活。工具对象内部持有对 client 传输层的引用，
      一旦 client 被 GC，工具调用将因连接关闭而失败。

    Args:
        config_file: 可选的 JSON 配置文件路径

    Returns:
        (tools: list, client: MultiServerMCPClient | None)
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print("[ToolRegistry] langchain_mcp_adapters 未安装，跳过 MCP 工具")
        return [], None

    servers_config, enabled_servers = (
        load_mcp_config(config_file) if config_file
        else (MCP_SERVERS, ENABLED_SERVERS)
    )
    enabled_config = get_enabled_servers(servers_config, enabled_servers)

    if not enabled_config:
        print("[ToolRegistry] 没有启用的 MCP 服务器")
        return [], None

    try:
        print(f"[ToolRegistry] 连接 MCP 服务器: {list(enabled_config.keys())}")
        client = MultiServerMCPClient(enabled_config)
        tools = await client.get_tools()

        # 过滤禁用的工具
        filtered = [t for t in tools if ENABLED_TOOLS.get(t.name, True)]

        print(f"[ToolRegistry] MCP 加载了 {len(filtered)} 个工具")
        for t in filtered:
            print(f"  - {t.name}: {t.description[:80]}")

        return filtered, client

    except Exception as e:
        print(f"[ToolRegistry] MCP 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return [], None


# ============================================================
# 工具注册中心
# ============================================================

class ToolRegistry:
    """
    工具注册中心 —— Agent 所有工具的单一入口。

    使用方式:
        registry = ToolRegistry()
        await registry.initialize(use_mcp=True, retriever=my_retriever)
        tools = registry.get_all_tools()  # → 传给 build_agent_graph

    Attributes:
        _mcp_tools:   MCP 远程工具列表（优先级最高）
        _rag_tool:    RAG 检索工具（可选）
        _local_tools: 本地内置工具列表（兜底）
        _mcp_client:  MCP 客户端实例（需保持存活）
    """

    def __init__(self):
        self._mcp_tools: list = []
        self._rag_tool: Optional[StructuredTool] = None
        self._local_tools: list = list(LOCAL_TOOLS)
        self._mcp_client = None

    async def initialize(
        self,
        use_mcp: bool = True,
        mcp_config_file: str = None,
        retriever=None,
    ):
        """
        初始化工具注册中心，按优先级加载三层工具。

        Args:
            use_mcp:         是否加载 MCP 远程工具
            mcp_config_file: MCP 配置文件路径
            retriever:       RAG 检索器（可选）
        """
        # 1. 注册 RAG 检索工具
        if retriever is not None:
            self._rag_tool = create_rag_tool(retriever)
            if self._rag_tool:
                print("[ToolRegistry] RAG 工具已注册: search_knowledge_base")

        # 2. 加载 MCP 远程工具
        if use_mcp:
            mcp_tools, client = await load_mcp_tools(mcp_config_file)
            if mcp_tools:
                self._mcp_tools = mcp_tools
                self._mcp_client = client

        # 3. 本地工具始终就绪
        print(f"[ToolRegistry] 本地工具已就绪: {[t.name for t in self._local_tools]}")

    def get_all_tools(self) -> list:
        """
        返回最终工具列表，自动去重。

        去重规则:
          MCP 工具 > 本地工具（同名的只保留 MCP 版本）

        Returns:
            list: 所有可用工具的列表
        """
        mcp_names = {t.name for t in self._mcp_tools}

        tools: list = []
        tools.extend(self._mcp_tools)  # 最高优先级

        if self._rag_tool:
            tools.append(self._rag_tool)

        # 本地工具去重: 排除已被 MCP 覆盖的同名工具
        for t in self._local_tools:
            if t.name not in mcp_names:
                tools.append(t)

        return tools

    @property
    def mcp_client(self):
        """
        MCP 客户端引用。

        调用方（如 main.py）必须在 Agent 使用期间持有此引用，
        防止 client 被 GC 导致工具调用失败。
        """
        return self._mcp_client

    @property
    def tool_count(self) -> int:
        """当前工具总数"""
        return len(self.get_all_tools())

    @property
    def tool_names(self) -> list:
        """当前所有工具的名称列表"""
        return [t.name for t in self.get_all_tools()]
