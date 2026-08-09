import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = BASE_DIR / "vector_stores" / "chroma_db"

# ========== 本地模型配置 ==========
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-0.5b-instruct:latest")
# 嵌入模型：默认使用 Ollama 的 bge-m3（免费、中文效果好、CPU 可跑）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "bge-m3")

# ========== 智谱 AI 免费模型配置 ==========
USE_LOCAL_LLM = False  # 关闭本地 LLM（使用智谱云）
USE_LOCAL_EMBEDDING = True  # 开启本地 embedding（Ollama bge-m3，免费）
USE_ZHIPU = True      # 开启智谱免费模型
# 智谱 AI 密钥（去官网免费拿）
ZHIPU_API_KEY = "dacc77c57e2e4e5ab307dd010c52d3c9.SgfGeAcMRfBAG1dT"
# 免费模型名称（固定不要改）
ZHIPU_MODEL_NAME = "glm-4.7-flash"
# 嵌入模型（免费、轻量、中文强） "embedding-2" 不免费，他们的
ZHIPU_EMBEDDING_MODEL = "embedding-2"


# 关闭 Chroma 遥测，消除错误日志
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_DEBUG"] = "false"

# RAG参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 3
TEMPERATURE = 0.0
# ========== RAG 检索总开关 ==========
# False: 完全跳过向量库加载，不调用 embedding API，Agent 只做纯 LLM 对话
# True:  加载向量库并将 search_knowledge_base 注册为 Agent 工具
ENABLE_RAG_RETRIEVAL = False
# 向量化开关：是否允许在运行时创建/更新向量数据库
# 仅 ENABLE_RAG_RETRIEVAL=True 时生效
ENABLE_VECTORIZE = False
# 是否强制重建向量库（删除已有持久化目录并重新创建）
FORCE_REBUILD_VECTORSTORE = False

# ========== Redis 配置（LangGraph Checkpointer 持久化）==========
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.4.60")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis88.7@")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
# LangGraph checkpoint 键前缀（多服务共享同一 Redis 时用于隔离）
REDIS_CHECKPOINT_PREFIX = os.getenv("REDIS_CHECKPOINT_PREFIX", "langgraph:checkpoint")