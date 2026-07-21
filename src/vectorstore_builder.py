from pathlib import Path
import json
import time
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config.settings import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, \
    USE_LOCAL_LLM, LOCAL_EMBEDDING_MODEL, ZHIPU_API_KEY, ZHIPU_EMBEDDING_MODEL
from src.loaders import load_documents_from_directory


def build_vectorstore(raw_data_dir=None, persist_dir=None, enable_vectorize=True, force_rebuild=False,
                      chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=None):
    """根据配置创建或加载向量数据库（embeddings 由内部按需创建）。

    Args:
        raw_data_dir: 原始文档目录。
        persist_dir: 向量库持久化目录（Path 或 str）。
        enable_vectorize: 是否允许在没有持久化库时进行向量化并创建库。
        force_rebuild: 如果为 True，则删除已有库并重新构建。
        chunk_size: 文本分块大小。
        chunk_overlap: 文本分块重叠长度。
        separators: 文本分块分隔符列表。

    Returns:
        vectorstore 实例（Chroma）。
    """
    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR
    if persist_dir is None:
        raise ValueError("persist_dir 不能为空")

    persist_path = Path(persist_dir)
    meta_path = persist_path / ".vectorstore_meta.json"

    # ── 仅在 ENABLE_VECTORIZE == True 且有实际向量化工作时才创建 ──
    _embeddings_cache = None

    def _create_embeddings():
        """按需创建 embeddings（缓存），查询和向量化均需要。"""
        nonlocal _embeddings_cache
        if _embeddings_cache is not None:
            return _embeddings_cache
        if USE_LOCAL_LLM:
            from langchain_ollama import OllamaEmbeddings
            _embeddings_cache = OllamaEmbeddings(model=LOCAL_EMBEDDING_MODEL)
        else:
            from zhipuai import ZhipuAI
            class _SafeZhipuEmbeddings:
                def __init__(self, api_key: str, model: str = "embedding-2"):
                    self.client = ZhipuAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
                    self.model = model
                def _clean_text(self, text: str) -> str:
                    if not text or not isinstance(text, str):
                        return ""
                    return text.strip()[:1500]
                def embed_documents(self, texts):
                    safe_texts = [self._clean_text(t) for t in texts if self._clean_text(t)]
                    if not safe_texts:
                        return []
                    batch_size = 32
                    all_embeddings = []
                    for i in range(0, len(safe_texts), batch_size):
                        batch = safe_texts[i:i+batch_size]
                        resp = self.client.embeddings.create(model=self.model, input=batch)
                        all_embeddings.extend([item.embedding for item in resp.data])
                    return all_embeddings
                def embed_query(self, text):
                    cleaned = self._clean_text(text)
                    if not cleaned:
                        raise ValueError("查询文本为空")
                    resp = self.client.embeddings.create(model=self.model, input=[cleaned])
                    return resp.data[0].embedding
            _embeddings_cache = _SafeZhipuEmbeddings(api_key=ZHIPU_API_KEY, model=ZHIPU_EMBEDDING_MODEL)
        return _embeddings_cache

    def _load_checkpoint():
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载向量库元数据失败：{e}")
            return {}

    def _save_checkpoint(timestamp, files):
        try:
            persist_path.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_indexed_at": timestamp,
                    "files": files,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存向量库元数据失败：{e}")

    def _scan_raw_files(raw_data_dir):
        raw_dir = Path(raw_data_dir)
        if not raw_dir.exists():
            return {}
        file_mtimes = {}
        for file_path in sorted(raw_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt", ".md"}:
                file_mtimes[str(file_path)] = file_path.stat().st_mtime
        return file_mtimes

    def _find_changed_files(meta):
        old_files = meta.get("files", {})
        current_files = _scan_raw_files(raw_data_dir)
        if not current_files:
            return [], [], current_files
        added_or_modified = []
        deleted = []
        for file_path, mtime in current_files.items():
            if file_path not in old_files or old_files[file_path] != mtime:
                added_or_modified.append(file_path)
        for file_path in old_files:
            if file_path not in current_files:
                deleted.append(file_path)
        return added_or_modified, deleted, current_files

    def _load_documents(changed_files=None):
        """载入文档，若指定 changed_files 则仅载入变更的文件（增量模式）。"""
        documents = load_documents_from_directory(raw_data_dir)
        if changed_files is not None:
            changed_set = set(changed_files)
            documents = [doc for doc in documents if doc.metadata.get("source") in changed_set]
        return documents or []

    def _split_documents(documents):
        nonlocal separators
        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        return text_splitter.split_documents(documents)

    if not enable_vectorize:
        if persist_path.exists() and any(persist_path.iterdir()):
            print("向量化被禁用；加载已有向量数据库")
            return Chroma(persist_directory=str(persist_path), embedding_function=_create_embeddings())
        raise RuntimeError(
            "向量化已被禁用，且未找到持久化向量数据库。请启用向量化或先构建向量库。"
        )

    if persist_path.exists() and any(persist_path.iterdir()) and not force_rebuild:
        checkpoint = _load_checkpoint()
        added_or_modified, deleted, current_files = _find_changed_files(checkpoint)

        if not added_or_modified and not deleted:
            print("未检测到新的或变更的文档，直接加载现有向量数据库")
            return Chroma(persist_directory=str(persist_path), embedding_function=_create_embeddings())

        # ── 有文件删除 → 全量重建 ──
        if deleted:
            print(f"检测到 {len(deleted)} 个文件被删除，将全量重建向量数据库：")
            for f in deleted:
                print(f"  - (已删除) {f}")
            for f in added_or_modified:
                print(f"  - {f}")
        else:
            # ── 仅有新增/修改 → 增量更新 ──
            print(f"检测到 {len(added_or_modified)} 个变更文件，进行增量更新：")
            for f in added_or_modified:
                print(f"  - {f}")

            print("正在加载已有向量数据库...")
            vectorstore = Chroma(persist_directory=str(persist_path), embedding_function=_create_embeddings())

            # 仅加载变更文件
            documents = _load_documents(changed_files=added_or_modified)
            if documents:
                chunks = _split_documents(documents)
                if chunks:
                    print(f"正在增量添加 {len(chunks)} 个文本块...")
                    vectorstore.add_documents(chunks)
                else:
                    print("变更文件分块后为空，跳过添加")
            else:
                print("变更文件中未提取到可处理的文档内容")

            _save_checkpoint(time.time(), current_files)
            print("向量数据库增量更新完成")
            return vectorstore

    elif force_rebuild:
        print("已开启强制重建，将重新构建向量数据库")
    else:
        print("未找到现有向量数据库，准备创建新的向量库")

    if persist_path.exists() and any(persist_path.iterdir()) and force_rebuild:
        print("检测到已有向量数据库，已设置强制重建，正在删除旧库...")
        try:
            shutil.rmtree(persist_path)
        except Exception as e:
            print(f"删除旧向量库失败：{e}")

    print("正在创建向量数据库...")
    embeddings = _create_embeddings()

    documents = _load_documents()
    if not documents:
        raise RuntimeError(f"在 {raw_data_dir} 中没有找到任何可处理的文档")
    chunks = _split_documents(documents)
    if not chunks:
        raise RuntimeError("文档分块后为空，无法创建向量库")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    _save_checkpoint(time.time(), _scan_raw_files(raw_data_dir))
    print("向量数据库创建完成")
    return vectorstore
