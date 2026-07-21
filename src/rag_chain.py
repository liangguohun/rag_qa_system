# 修复：文本分割器新路径
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ✅ 新版 Ollama 包（无废弃警告）
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 新版 RAG 链核心导入
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI

from zhipuai import ZhipuAI  # 官方SDK
import json


# mcp 接入修改
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.mcp_config import MCP_SERVERS, ENABLED_SERVERS, load_mcp_config, get_enabled_servers, ENABLED_TOOLS
from pathlib import Path
import sys
ROOT = Path(__file__).parent.parent
# from langchain.agents import create_agent

from config.settings import *
from src.vectorstore_builder import build_vectorstore
import requests
import os
import shutil

# 写死tools 的写法用例
'''

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------- 工具封装成 Skill 模块 ----------------------
# ---------------------------------------------------------------------------

# ---------------------- 新增简单 Tool 用例 ----------------------
# 新增 tool 相关导入
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain.agents import create_react_agent, AgentExecutor
# 不需要 AgentExecutor，create_agent 返回的对象可以直接调用 .invoke()
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
import math  # 别忘了导入 math
import json

@tool
def get_current_time(format_type: str = "full") -> str:
    """
    获取当前时间。
    
    Args:
        format_type: 时间格式，可选值：
            - "full": 完整格式（YYYY-MM-DD HH:MM:SS）
            - "date": 仅日期（YYYY-MM-DD）
            - "time": 仅时间（HH:MM:SS）
    
    Returns:
        格式化的当前时间字符串
    """
    now = datetime.datetime.now()
    
    if format_type == "date":
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        return now.strftime("%H:%M:%S")
    else:  # full
        return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。
    
    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * 4" 或 "10 / 2"
    
    Returns:
        计算结果字符串
    """
    try:
        # 安全地计算表达式（注意：生产环境需要更严格的限制）
        # 只允许基本数学运算
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        
        # 编译并计算
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"不允许使用 {name} 函数")
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_word_length(word: str) -> str:
    """
    获取单词或文本的长度。
    
    Args:
        word: 需要计算长度的字符串
    
    Returns:
        字符串长度信息
    """
    length = len(word)
    return f"'{word}' 的长度是 {length} 个字符"

@tool
def weather_check(city: str = "北京") -> str:
    """
    查询城市天气（模拟示例）。
    
    Args:
        city: 城市名称，默认为北京
    
    Returns:
        模拟的天气信息
    """
    # 这是一个模拟的天气函数，实际使用时需要接入真实API
    weather_data = {
        "北京": "晴天，温度 25°C，湿度 45%",
        "上海": "多云，温度 28°C，湿度 60%",
        "广州": "小雨，温度 30°C，湿度 75%",
        "深圳": "阴天，温度 29°C，湿度 70%",
    }
    
    weather_info = weather_data.get(city, f"{city}：晴天，温度 22°C，湿度 50%")
    return f"{city}天气：{weather_info}"
# 统一 Skill 列表
ALL_SKILLS = [
    get_current_time,
    calculate,
    get_word_length,
    weather_check
]

# ---------------------------------------------------------------------------
# ---------------------- Agent 封装（稳定调用版） ----------------------
# ---------------------------------------------------------------------------
    
# ---------------------- 创建带工具的 Agent ----------------------
def create_agent_with_tools(llm):

    """创建带有工具的 Agent"""
    # # 定义工具列表
    # tools = [get_current_time, calculate, get_word_length, weather_check]
    
    """
    封装为 Skill 助手
    稳定适配 langchain==1.2.15 + 智谱 GLM
    """
    tools = ALL_SKILLS
    # 绑定工具 schema（必须加，否则工具不触发）
    llm_with_tools = llm.bind_tools(tools)

    # 创建 Agent 提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
            """
            你是一个有帮助的助手，可以使用以下工具来回答用户问题：
            {tools}

            工具名称: {tool_names}

            使用说明：
            1. 如果需要获取当前时间，使用 get_current_time 工具
            2. 如果需要计算数学表达式，使用 calculate 工具
            3. 如果需要获取文本长度，使用 get_word_length 工具
            4. 如果需要查询天气，使用 weather_check 工具

            回答时请基于工具返回的结果，以友好的方式呈现给用户。
            """
         ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 只传字符串！不传模板！
    system_prompt = """
        你是一个有帮助的助手，可以使用以下工具来回答用户问题：
        {tools}

        工具名称: {tool_names}

        使用说明：
        1. 需要当前时间 → get_current_time
        2. 需要数学计算 → calculate
        3. 需要文本长度 → get_word_length
        4. 需要查询天气 → weather_check

        必须根据问题选择合适的工具，不能编造答案。
        """

    # # 创建 Agent 用这个方式会调用，但不是很准
    # agent = create_agent(
    #     model=llm, 
    #     tools=tools,
    #     # system_prompt=prompt,
    #     system_prompt=system_prompt,
    #     # verbose=True
    # )
    
    # 创建 Agent 用这个方式会调用，但不是很准
    agent = create_agent(
        llm_with_tools,
        tools=tools,
        system_prompt=system_prompt
    )

    # agent = create_tool_calling_agent(llm, tools, prompt)
    # # 创建 Agent Executor
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=True,  # 打印详细执行过程
    #     handle_parsing_errors=True,
    #     max_iterations=5,
    # )

    return agent, tools

# ---------------------- 简单的 Tool 测试函数 ----------------------
def test_tools():
    """测试各个工具的功能"""
    print("\n" + "="*50)
    print("测试简单工具")
    print("="*50)
    
    # 测试 1: 获取当前时间
    print("\n1. 测试 get_current_time:")
    print(f"   full格式: {get_current_time.invoke({'format_type': 'full'})}")
    print(f"   date格式: {get_current_time.invoke({'format_type': 'date'})}")
    print(f"   time格式: {get_current_time.invoke({'format_type': 'time'})}")
    
    # 测试 2: 计算器
    print("\n2. 测试 calculate:")
    print(f"   {calculate.invoke({'expression': '2 + 3 * 4'})}")
    print(f"   {calculate.invoke({'expression': '(10 + 5) / 3'})}")
    print(f"   {calculate.invoke({'expression': '2 ** 3'})}")
    
    # 测试 3: 文本长度
    print("\n3. 测试 get_word_length:")
    print(f"   {get_word_length.invoke({'word': 'Hello World'})}")
    print(f"   {get_word_length.invoke({'word': 'LangChain 学习'})}")
    
    # 测试 4: 天气查询
    print("\n4. 测试 weather_check:")
    print(f"   {weather_check.invoke({'city': '北京'})}")
    print(f"   {weather_check.invoke({'city': '上海'})}")
    print(f"   {weather_check.invoke({'city': '未知城市'})}")
'''


# ---------------------- MCP工具加载（可配置版）-----------------------
async def load_mcp_tools(config_file: str = None):
    """
    从配置加载MCP工具（同时返回 client，调用方必须保持 client 存活）

    Args:
        config_file: 可选的配置文件路径

    Returns:
        (tools, client) 元组：MCP工具列表 + MultiServerMCPClient 实例
    """
    # 加载配置
    servers_config, enabled_servers = load_mcp_config(config_file) if config_file else (MCP_SERVERS, ENABLED_SERVERS)

    # 获取启用的服务器配置（过滤掉enabled字段）
    enabled_servers_config = get_enabled_servers(servers_config, enabled_servers)

    if not enabled_servers_config:
        print("⚠️ 没有启用的MCP服务器")
        return [], None

    try:
        print(f"📡 连接到MCP服务器: {list(enabled_servers_config.keys())}")

        # 创建MCP客户端（只传入支持的配置）
        client = MultiServerMCPClient(enabled_servers_config)
        tools = await client.get_tools()

        # 根据配置过滤工具
        filtered_tools = []
        for tool in tools:
            if ENABLED_TOOLS.get(tool.name, True):
                filtered_tools.append(tool)
            else:
                print(f"⏭️ 跳过禁用的工具: {tool.name}")

        print(f"✅ MCP加载了 {len(filtered_tools)} 个工具")
        for tool in filtered_tools:
            print(f"   - {tool.name}: {tool.description}")
            print(f"     [诊断] type={type(tool).__name__}, has_ainvoke={hasattr(tool, 'ainvoke')}, has_invoke={hasattr(tool, 'invoke')}")
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    print(f"     [诊断] args_schema={tool.args_schema.schema()}")
                except Exception:
                    print(f"     [诊断] args_schema={tool.args_schema}")

        return filtered_tools, client
    except Exception as e:
        print(f"❌ MCP工具加载失败: {e}")
        import traceback
        traceback.print_exc()
        return [], None

from langchain.agents import create_agent 

# ---------------------- 创建带工具的Agent（支持MCP）-----------------------
async def create_agent_with_tools(llm, use_mcp: bool = True, mcp_config_file: str = None, retriever=None):
    """
    创建带有工具的Agent

    Args:
        llm: 语言模型
        use_mcp: 是否使用MCP工具
        mcp_config_file: MCP配置文件路径
        retriever: 可选的 RAG 检索器，用于知识库检索增强

    Returns:
        (agent, tools, mcp_client) 元组。mcp_client 必须保持存活以确保工具可用。
    """
    all_tools = []
    filtered_local_tools = []
    mcp_tools = []
    mcp_client = None

    # 0. 如果提供了 retriever，封装为 RAG 检索工具
    if retriever is not None:
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def search_knowledge_base(query: str) -> str:
            """
            在知识库中检索相关文档内容。当需要查找事实、概念、文档中的具体信息时使用。
            Args:
                query: 检索查询词
            Returns:
                检索到的相关文档内容
            """
            docs = retriever.invoke(query)
            if not docs:
                return "知识库中未找到相关信息"
            return "\n\n---\n\n".join(
                f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
                for doc in docs
            )

        all_tools.append(search_knowledge_base)
        print("✅ RAG 检索工具已注入 Agent")

    # 1. 如果启用 MCP，则优先加载 MCP 工具
    if use_mcp:
        try:
            mcp_tools, mcp_client = await load_mcp_tools(mcp_config_file)
            if mcp_tools:
                all_tools.extend(mcp_tools)
            print(f"✅ MCP工具已加载: {len(mcp_tools)} 个工具")
        except Exception as e:
            print(f"⚠️ MCP工具加载失败: {e}，将尝试加载本地工具")

    # 2. 如果没有加载到 MCP 工具，则回退到本地工具
    if not all_tools:
        print("⚠️ 未加载到 MCP 工具，使用本地工具作为后备")
        local_tools = ALL_SKILLS
        filtered_local_tools = [
            tool for tool in local_tools
            if ENABLED_TOOLS.get(tool.name, True)
        ]
        all_tools.extend(filtered_local_tools)
        print(f"✅ 本地工具已加载: {len(filtered_local_tools)} 个工具")

    # 3. 如果仍然没有工具，则返回失败
    if not all_tools:
        print("⚠️ 没有可用的工具，Agent将无法调用工具")
        return None, [], mcp_client

    system_prompt = """
                你是一个严格执行指令的助手，只能使用工具返回的精确数据回答问题。

                工具名称：{tool_names}

                核心规则（必须严格遵守）：
                1. 收到工具返回结果后，必须原样呈现其中的数据，**严禁修改、猜测或编造任何数字、名称和事实**。
                2. 例如：工具返回"25°C"，你不能说"18°C"或"大约20°C"。
                3. 如果工具返回的内容已经完整，直接输出原文即可，无需"润色"。
                4. 如果问题涉及文档查找，优先使用 search_knowledge_base 检索。
            """

    # 尝试用 bind_tools 绑定工具，尤其是 MCP StructuredTool 需要 async 支持
    llm_with_tools = None
    if hasattr(llm, "bind_tools"):
        try:
            llm_with_tools = llm.bind_tools(all_tools)
            print("✅ LLM bind_tools 成功，可支持 StructuredTool 异步调用")
        except Exception as e:
            print(f"⚠️ LLM bind_tools 失败：{e}")

    agent_model = llm_with_tools if llm_with_tools is not None else llm

    print(f"\n[诊断] 创建 Agent，传入工具共 {len(all_tools)} 个:")
    for t in all_tools:
        print(f"  - name={t.name}, type={type(t).__name__}, has_ainvoke={hasattr(t, 'ainvoke')}, has_invoke={hasattr(t, 'invoke')}")

    agent = create_agent(
        llm_with_tools,
        tools=all_tools,
        system_prompt=system_prompt
    )
    print(f"[诊断] Agent 创建完成，type={type(agent).__name__}")
    return agent, all_tools, mcp_client

# ---------------------- 异步MCP工具测试 ----------------------
async def test_mcp_tools_async(config_file: str = None):
    """异步测试MCP工具"""
    print("\n" + "="*50)
    print("测试MCP工具（异步）")
    print("="*50)
    
    tools, _ = await load_mcp_tools(config_file)
    
    if not tools:
        print("没有加载到MCP工具")
        return
    
    # 测试第一个工具
    if tools:
        print(f"\n测试工具: {tools[0].name}")
        try:
            # 根据工具名称调用不同的参数
            if tools[0].name == "get_current_time":
                result = await tools[0].ainvoke({"format_type": "full"})
            else:
                result = await tools[0].ainvoke({})
            print(f"结果: {result}")
        except Exception as e:
            print(f"调用失败: {e}")

def create_rag_chain():
    """创建 RAG 链（LangChain v1.0+ 无警告版）"""

    # 1. 准备 LLM（embeddings 由 build_vectorstore 内部按需创建）
    if USE_LOCAL_LLM:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=TEMPERATURE)
    else:
        llm = ChatOpenAI(
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL_NAME,
            temperature=TEMPERATURE,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

    # 2. 创建或加载向量库（build_vectorstore 仅在 ENABLE_VECTORIZE=True 且有变更/新文件时创建 embeddings）
    vectorstore = build_vectorstore(
        raw_data_dir=RAW_DATA_DIR,
        persist_dir=VECTOR_STORE_DIR,
        enable_vectorize=ENABLE_VECTORIZE,
        force_rebuild=FORCE_REBUILD_VECTORSTORE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # ===================== 新版 RAG 链 =====================
    prompt = PromptTemplate.from_template("""
                使用以下上下文来回答最后的问题。
                如果你不知道答案，就直接说不知道，不要编造答案。

                上下文：
                {context}

                问题：{question}

                回答：
                """
            )

    def format_docs(docs):
        if not docs:
            return "没有找到相关上下文"
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    # ======================================================
    
    return rag_chain, retriever, llm  # 返回 llm 以便创建 agent