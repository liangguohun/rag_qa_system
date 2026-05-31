from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from pathlib import Path
from typing import List
# 新版正确导入路径
from langchain_core.documents import Document

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """加载目录中的所有文档"""
    docs = []
    path = Path(directory_path)
    
    for file_path in path.iterdir():
        if file_path.suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
        elif file_path.suffix == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs.extend(loader.load())
        elif file_path.suffix == '.md':
            loader = UnstructuredMarkdownLoader(str(file_path))
            docs.extend(loader.load())
    
    return docs