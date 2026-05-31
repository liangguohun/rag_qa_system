#!/usr/bin/env python
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import load_documents_from_directory
from src.rag_chain import create_rag_chain
from config.settings import RAW_DATA_DIR

def main():
    print(f"从 {RAW_DATA_DIR} 加载文档...")
    docs = load_documents_from_directory(RAW_DATA_DIR)
    print(f"加载了 {len(docs)} 个文档片段")
    
    print("创建向量数据库...")
    # 这会触发重新创建向量库
    create_rag_chain()
    print("完成！")

if __name__ == "__main__":
    main()