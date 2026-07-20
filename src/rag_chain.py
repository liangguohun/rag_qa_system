# 修复：文本分割器新路径
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

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


# mcp 接入修改
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.mcp_config import MCP_SERVERS, ENABLED_SERVERS, load_mcp_config, get_enabled_servers, ENABLED_TOOLS
from pathlib import Path
import sys
ROOT = Path(__file__).parent.parent
# from langchain.agents import create_agent

from config.settings import *
import requests
import os
import shutil

# ---------------------- 修复版的智谱 Embedding ----------------------
class SafeZhipuEmbeddings:
    def __init__(self, api_key: str, model: str = "embedding-2"):
        self.client = ZhipuAI(api_key=api_key, base_url = "https://open.bigmodel.cn/api/paas/v4/")
        self.model = model
    
    def _clean_text(self, text: str) -> str:
        """清理和截断文本"""
        if not text or not isinstance(text, str):
            return ""
        # 去除首尾空白，限制长度（embedding-2 最大 512 tokens，约 1000-1500 字符）
        cleaned = text.strip()[:1500]
        return cleaned
    
    def embed_documents(self, texts):
        """批量向量化文档"""
        # 过滤空文本并清理
        safe_texts = []
        for t in texts:
            cleaned = self._clean_text(t)
            if cleaned:  # 只添加非空文本
                safe_texts.append(cleaned)
        
        if not safe_texts:
            print("警告：没有有效的文本需要向量化")
            return []
        
        try:
            # 批量处理，每次最多 32 条（API 限制）
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(safe_texts), batch_size):
                batch = safe_texts[i:i+batch_size]
                print(f"正在向量化第 {i//batch_size + 1} 批，共 {len(batch)} 条文本")
                
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # 提取 embeddings，确保顺序正确
                batch_embeddings = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_embeddings)
            
            print(f"成功向量化 {len(all_embeddings)} 条文本")
            return all_embeddings
            
        except Exception as e:
            print(f"Embedding调用失败：{e}")
            # 打印更多错误信息用于调试
            if hasattr(e, 'response'):
                print(f"响应内容：{e.response.text}")
            raise
    
    def embed_query(self, text):
        """向量化单个查询"""
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("查询文本为空")
        
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=[cleaned]  # 注意：必须是列表格式
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"查询向量化失败：{e}")
            raise


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
    从配置加载MCP工具
    
    Args:
        config_file: 可选的配置文件路径
    
    Returns:
        MCP工具列表
    """
    # 加载配置
    servers_config, enabled_servers = load_mcp_config(config_file) if config_file else (MCP_SERVERS, ENABLED_SERVERS)
    
    # 获取启用的服务器配置（过滤掉enabled字段）
    enabled_servers_config = get_enabled_servers(servers_config, enabled_servers)
    
    if not enabled_servers_config:
        print("⚠️ 没有启用的MCP服务器")
        return []
    
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
        
        return filtered_tools
    except Exception as e:
        print(f"❌ MCP工具加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []

from langchain.agents import create_agent 

# ---------------------- 创建带工具的Agent（支持MCP）-----------------------
def create_agent_with_tools(llm, use_mcp: bool = True, mcp_config_file: str = None ):
    """
    创建带有工具的Agent
    
    Args:
        llm: 语言模型
        use_mcp: 是否使用MCP工具
        mcp_config_file: MCP配置文件路径
        verbose: 是否打印详细执行过程
    """
    all_tools = []
    
    # # 1. 加载本地工具（始终加载）
    # local_tools = ALL_SKILLS
    
    # 根据配置过滤本地工具
    filtered_local_tools = [
        # tool for tool in local_tools 
        # if ENABLED_TOOLS.get(tool.name, True)
    ]
    all_tools.extend(filtered_local_tools)
    
    # 2. 加载MCP工具（可选）
    if use_mcp:
        try:
            # 同步方式加载MCP工具
            import asyncio
            mcp_tools = asyncio.run(load_mcp_tools(mcp_config_file))
            all_tools.extend(mcp_tools)
            print(f"✅ 共加载 {len(all_tools)} 个工具 (本地: {len(filtered_local_tools)}, MCP: {len(mcp_tools)})")
        except Exception as e:
            print(f"⚠️ MCP工具加载失败: {e}，仅使用本地工具")
    
    # 如果没有工具，返回提示
    if not all_tools:
        print("⚠️ 没有可用的工具，Agent将无法调用工具")
        return None, []
    
    ''' 
        # 创建Agent
        # 'OllamaLLM' object has no attribute 'bind_tools' 
        llm_with_tools = llm.bind_tools(all_tools)
        
        system_prompt = """
            你是一个有帮助的助手，可以使用以下工具来回答用户问题：
            {tools}

            工具名称: {tool_names}

            使用说明：
            1. 需要当前时间 → get_current_time
            2. 需要数学计算 → calculate
            3. 需要文本长度 → get_word_length
            4. 需要查询天气 → weather_check
            5. 其他工具请根据名称和描述选择合适的工具

            必须根据问题选择合适的工具，不能编造答案。
        """
        
        agent = create_agent(
            llm_with_tools,
            tools=all_tools,
            system_prompt=system_prompt
        )
        
        return agent, all_tools
    '''
   
    if not all_tools:
        print("⚠️ 没有可用的工具")
        return None

    system_prompt = """
                你是一个有帮助的助手，可以使用以下工具来回答用户问题：
                {all_tools}

                工具名称: {tool_names}

                使用说明：
                1. 需要当前时间 → get_current_time
                2. 需要数学计算 → calculate
                3. 需要文本长度 → get_word_length
                4. 需要查询天气 → weather_check
                5. 其他工具请根据名称和描述选择合适的工具

                必须根据问题选择合适的工具，不能编造答案。
            """
            
    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=system_prompt
    )
    return agent

# ---------------------- 异步MCP工具测试 ----------------------
async def test_mcp_tools_async(config_file: str = None):
    """异步测试MCP工具"""
    print("\n" + "="*50)
    print("测试MCP工具（异步）")
    print("="*50)
    
    tools = await load_mcp_tools(config_file)
    
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


def build_vectorstore(chunks, embeddings=None, persist_dir=None, enable_vectorize=True, force_rebuild=False):
    """
    根据配置创建或加载向量数据库。

    Args:
        chunks: 要索引的文档块列表
        embeddings: embedding 对象（需兼容 Chroma API）
        persist_dir: 向量库持久化目录（Path 或 str）
        enable_vectorize: 是否允许在没有持久化库时进行向量化并创建库
        force_rebuild: 如果为 True，则删除已有库并重新构建

    Returns:
        vectorstore 实例（Chroma）
    """
    from pathlib import Path

    if persist_dir is None:
        raise ValueError("persist_dir 不能为空")
    persist_path = Path(persist_dir)

    # 如果不允许向量化，只能尝试加载已有持久化库
    if not enable_vectorize:
        if persist_path.exists() and any(persist_path.iterdir()):
            print("向量化被禁用；加载已有向量数据库")
            # 尝试不传 embedding（仅加载已持久化的数据库）
            try:
                return Chroma(persist_directory=str(persist_path))
            except TypeError:
                # 兼容旧版 API
                return Chroma(persist_directory=str(persist_path), embedding=embeddings)
        else:
            raise RuntimeError(
                "向量化已被禁用，且未找到持久化向量数据库。请启用向量化或先构建向量库。"
            )

    # enable_vectorize == True: 根据是否存在库和是否强制重建决定行为
    if persist_path.exists() and any(persist_path.iterdir()):
        if force_rebuild:
            print("检测到已有向量数据库，已设置强制重建，正在删除旧库...")
            try:
                shutil.rmtree(persist_path)
            except Exception as e:
                print(f"删除旧向量库失败：{e}")
        else:
            print("检测到已有向量数据库，直接加载（未强制重建）")
            try:
                return Chroma(persist_directory=str(persist_path))
            except TypeError:
                return Chroma(persist_directory=str(persist_path), embedding=embeddings)

    # 创建新的向量库
    print("正在创建向量数据库...")
    if embeddings is None:
        raise RuntimeError("要创建新的向量库，必须提供 embeddings 实例")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path)
    )
    print("向量数据库创建完成")
    return vectorstore

def create_rag_chain():
    """创建 RAG 链（LangChain v1.0+ 无警告版）"""
    
    # 1. 加载文档
    from src.loaders import load_documents_from_directory
    documents = load_documents_from_directory(RAW_DATA_DIR)
    
    # 检查文档是否为空
    if not documents:
        raise ValueError(f"在 {RAW_DATA_DIR} 中没有找到任何文档")
    
    print(f"加载了 {len(documents)} 个文档")
    
    # 2. 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise ValueError("文档分块后为空")
    
    print(f"文档分块为 {len(chunks)} 个块")
    
    # 3. 向量化（在 ENABLE_VECTORIZE 控制下创建 embeddings，避免不必要的 token 消耗）
    embeddings = None
    if ENABLE_VECTORIZE:
        if USE_LOCAL_LLM:
            # ✅ 新版 Ollama 嵌入模型（无废弃警告）
            embeddings = OllamaEmbeddings(model=LOCAL_EMBEDDING_MODEL)
            # 本地模型 - 使用Ollama但不支持bind_tools
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=LOCAL_LLM_MODEL,
                temperature=TEMPERATURE,
            )
        else:
            # 使用修复后的智谱 Embedding（实际向量化在 Chroma.from_documents 时触发）
            embeddings = SafeZhipuEmbeddings(
                api_key=ZHIPU_API_KEY,
                model=ZHIPU_EMBEDDING_MODEL
            )
            llm = ChatOpenAI(
                api_key=ZHIPU_API_KEY,
                model=ZHIPU_MODEL_NAME,
                temperature=TEMPERATURE,
                base_url="https://open.bigmodel.cn/api/paas/v4/",
            )
    else:
        # 不进行向量化时仍需创建 LLM 用于生成（但避免创建 embeddings）
        if USE_LOCAL_LLM:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model=LOCAL_LLM_MODEL,
                temperature=TEMPERATURE,
            )
        else:
            llm = ChatOpenAI(
                api_key=ZHIPU_API_KEY,
                model=ZHIPU_MODEL_NAME,
                temperature=TEMPERATURE,
                base_url="https://open.bigmodel.cn/api/paas/v4/",
            )

        # # 1. 测试网络连通性
        # try:
        #     response = requests.get("https://open.bigmodel.cn/api/paas/v4/", timeout=5)
        #     print(f"网络连通性: {response.status_code}")
        # except Exception as e:
        #     print(f"网络错误: {e}")

        # # 2. 测试SSL证书
        # try:
        #     response = requests.get("https://open.bigmodel.cn", verify=True)
        #     print("SSL证书验证通过")
        # except Exception as e:
        #     print(f"SSL错误: {e}")
        #     print("尝试跳过SSL验证...")
        #     response = requests.get("https://open.bigmodel.cn", verify=False)
        #     print("跳过SSL验证后连接成功")
    # 4. 创建或加载向量库（使用封装函数，受配置开关控制）
    vectorstore = build_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        persist_dir=VECTOR_STORE_DIR,
        enable_vectorize=ENABLE_VECTORIZE,
        force_rebuild=FORCE_REBUILD_VECTORSTORE,
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