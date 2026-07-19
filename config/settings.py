import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = BASE_DIR / "vector_stores" / "chroma_db"

# # 模型配置
# USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# # LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
# # LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
# LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
# # 嵌入模型（向量化用 → 必须是 Ollama 支持的模型） 或则 nomic-embed-text 中文友好
# LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "bge-m3")

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-0.5b-instruct:latest")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
# Ollama 的 embedding 模型 nomic-embed-text英文效果好
# LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")

# ========== 智谱 AI 免费模型配置 ==========
USE_LOCAL_LLM = False  # 关闭本地模型
USE_ZHIPU = True      # 开启智谱免费模型
# 智谱 AI 密钥（去官网免费拿）
ZHIPU_API_KEY = "dacc77c57e2e4e5ab307dd010c52d3c9.SgfGeAcMRfBAG1dT"
# 免费模型名称（固定不要改）
ZHIPU_MODEL_NAME = "glm-4-flash"
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