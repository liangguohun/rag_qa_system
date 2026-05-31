import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = BASE_DIR / "vector_stores" / "chroma_db"

# 模型配置
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
# LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
# 嵌入模型（向量化用 → 必须是 Ollama 支持的模型） 或则 nomic-embed-text 中文友好
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "bge-m3")

# 关闭 Chroma 遥测，消除错误日志
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_DEBUG"] = "false"

# RAG参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 3
TEMPERATURE = 0.0