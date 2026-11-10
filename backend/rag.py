from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import redis
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model import embeddings_model

# 显式加载 backend/.env，避免读到父项目 .env 污染 REDIS_URL
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
log = logging.getLogger(__name__)

RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME", "weather")
RAG_RECALL_K = int(os.getenv("RAG_RECALL_K", "40"))
RAG_TOP_N = int(os.getenv("RAG_TOP_N", "3"))
RAG_CHILD_SIZE = int(os.getenv("RAG_CHILD_SIZE", "500"))
RAG_CHILD_OVERLAP = int(os.getenv("RAG_CHILD_OVERLAP", "80"))
RAG_BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "20"))
RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
RAG_RERANK_THRESHOLD = os.getenv("RAG_RERANK_THRESHOLD", "").strip()
# 默认不设硬阈值，可显式指定 0~1
RAG_RERANK_THRESHOLD = (
    float(RAG_RERANK_THRESHOLD) if RAG_RERANK_THRESHOLD else None
)
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "10000"))
RAG_PARENT_MAX_CHARS = int(os.getenv("RAG_PARENT_MAX_CHARS", "3500"))
RAG_MAX_PAGES_PER_SOURCE = int(os.getenv("RAG_MAX_PAGES_PER_SOURCE", "2"))
# 默认禁用邻页扩展，启用时仅在子块贴近页边界时补充
RAG_NEIGHBOR_PAGES = int(os.getenv("RAG_NEIGHBOR_PAGES", "0"))
RAG_NEIGHBOR_CHARS = int(os.getenv("RAG_NEIGHBOR_CHARS", "300"))
RAG_EDGE_CHARS = int(os.getenv("RAG_EDGE_CHARS", "120"))

if not (RAG_CHILD_SIZE > RAG_CHILD_OVERLAP >= 0):
    raise ValueError("需满足 RAG_CHILD_SIZE > RAG_CHILD_OVERLAP >= 0")
if min(RAG_RECALL_K, RAG_TOP_N, RAG_BATCH_SIZE, RAG_MAX_CONTEXT_CHARS,
       RAG_PARENT_MAX_CHARS, RAG_MAX_PAGES_PER_SOURCE) < 1:
    raise ValueError("召回数、返回数、批量大小及上下文限制必须为正整数")
if RAG_RERANK_THRESHOLD is not None and not 0 <= RAG_RERANK_THRESHOLD <= 1:
    raise ValueError("RAG_RERANK_THRESHOLD 必须在 0~1 之间")


@lru_cache(maxsize=1)
def _get_redis():
    url = os.getenv("REDIS_URL")
    if not url:
        raise ValueError("缺少 REDIS_URL 环境变量")
    return redis.from_url(url, decode_responses=True)


def _parents_key(index_name: str = RAG_INDEX_NAME) -> str:
    return f"rag:{index_name}:parents"


def _manifest_key(source_id: str, index_name: str = RAG_INDEX_NAME) -> str:
    return f"rag:{index_name}:manifest:{source_id}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _splitter() -> RecursiveCharacterTextSplitter:
    # 中文段落 / 句子优先切分
    return RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHILD_SIZE,
        chunk_overlap=RAG_CHILD_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ";", " ", ""],
        length_function=len,
        add_start_index=True,
    )


@lru_cache(maxsize=8)
def get_vector_store(index_name: str = RAG_INDEX_NAME) -> RedisVectorStore:
    # 写入 / 查询共用索引配置，新索引用 HNSW
    return RedisVectorStore(
        embeddings=embeddings_model(),
        config=RedisConfig(
            index_name=index_name,
            redis_url=os.environ["REDIS_URL"],
            indexing_algorithm="HNSW",
        ),
    )


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RAG_RERANK_MODEL, local_files_only=True)


def _batch_delete(store: RedisVectorStore, ids: list[str]) -> None:
    for offset in range(0, len(ids), 200):
        store.delete(ids=ids[offset:offset + 200])


def add_rag_pdf(file_path: str = r"rag\9787502958572_L.pdf") -> dict[str, Any]:
    """导入 PDF；同版本跳过，更新时清理旧版本向量（中断后用 RAG_REBUILD=1 重建）。"""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    source = str(path)
    source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    version_payload = (
        _hash_file(path)
        + f"|page-zh-v2|{RAG_CHILD_SIZE}|{RAG_CHILD_OVERLAP}"
    )
    version = hashlib.sha256(version_payload.encode("utf-8")).hexdigest()[:16]
    document_id = f"{source_id}:{version}"
    client = _get_redis()
    manifest_key = _manifest_key(source_id)
    old_raw = client.get(manifest_key)
    old = json.loads(old_raw) if old_raw else None
    if old and old.get("version") == version:
        return {"source": source, "status": "skipped", "pages": len(old["parents"]),
                "children": len(old["children"])}

    documents = PyMuPDFLoader(file_path=source, mode="page").load()
    splitter = _splitter()
    parents: dict[str, str] = {}
    children: list[Document] = []
    child_ids: list[str] = []
    for page_doc in documents:
        page_text = page_doc.page_content.strip()
        if not page_text:
            continue
        page_index = int(page_doc.metadata["page"])  # PyMuPDFLoader: 0-based
        parent_id = f"{document_id}:p{page_index}"
        parents[parent_id] = page_text
        for i, child in enumerate(splitter.create_documents([page_text])):
            chunk_id = f"{document_id}:p{page_index}:c{i}"
            child.metadata.update({
                "parent_id": parent_id,
                "document_id": document_id,
                "source": source,
                "filename": path.name,
                "page": page_index + 1,  # 展示页码：1-based
                "page_index": page_index,
            })
            children.append(child)
            child_ids.append(chunk_id)

    if not children:
        return {"source": source, "status": "empty", "pages": 0, "children": 0}

    store = get_vector_store()
    # 先写父块，避免检索到新子块时父块仍不存在
    client.hset(_parents_key(), mapping=parents)
    for i in range(0, len(children), RAG_BATCH_SIZE):
        store.add_documents(
            children[i:i + RAG_BATCH_SIZE],
            ids=child_ids[i:i + RAG_BATCH_SIZE],
        )

    # 新版本可检索后再回收旧版本向量和父块
    if old:
        _batch_delete(store, old.get("children", []))
        old_parent_ids = old.get("parents", [])
        if old_parent_ids:
            client.hdel(_parents_key(), *old_parent_ids)

    client.set(manifest_key, json.dumps({
        "source": source,
        "version": version,
        "children": child_ids,
        "parents": list(parents),
    }, ensure_ascii=False))
    return {"source": source, "status": "indexed", "pages": len(parents),
            "children": len(children)}


def del_rag(index_name: str = RAG_INDEX_NAME) -> None:
    """删除当前索引、父块及导入清单（全量删除）。"""
    client = _get_redis()
    try:
        client.ft(index_name).dropindex(delete_documents=True)
    except redis.exceptions.ResponseError as exc:
        msg = str(exc).lower()
        if "unknown index" not in msg and "no such index" not in msg:
            raise
    client.delete(_parents_key(index_name))
    manifests = list(client.scan_iter(match=f"rag:{index_name}:manifest:*", count=100))
    if manifests:
        client.delete(*manifests)
    get_vector_store.cache_clear()


def _trim_page(text: str, hit_start: int, limit: int) -> str:
    """普通页返回完整原文；超长页围绕命中位置截取，避免耗尽上下文。"""
    if len(text) <= limit:
        return text
    left = max(0, min(hit_start - limit // 3, len(text) - limit))
    right = left + limit
    return ("……" if left else "") + text[left:right] + ("……" if right < len(text) else "")


def _neighbor_excerpt(
    client: Any, parent_key: str, selected_ids: set[str],
    document_id: str, page_index: int, hit_start: int, hit_length: int,
    page_length: int,
) -> str:
    if RAG_NEIGHBOR_PAGES <= 0 or RAG_NEIGHBOR_CHARS <= 0:
        return ""
    extra: list[str] = []
    for distance in range(1, RAG_NEIGHBOR_PAGES + 1):
        # 仅子块命中页首 / 页尾时补邻页，不代表语义相关
        for direction in (-1, 1):
            boundary_hit = (
                hit_start <= RAG_EDGE_CHARS if direction < 0 else
                hit_start + hit_length >= page_length - RAG_EDGE_CHARS
            )
            if not boundary_hit:
                continue
            neighbor_idx = page_index + distance * direction
            if neighbor_idx < 0:
                continue
            neighbor_id = f"{document_id}:p{neighbor_idx}"
            if neighbor_id in selected_ids:
                continue
            neighbor_text = client.hget(parent_key, neighbor_id)
            if not neighbor_text:
                continue
            excerpt = (
                neighbor_text[-RAG_NEIGHBOR_CHARS:]
                if direction < 0 else neighbor_text[:RAG_NEIGHBOR_CHARS]
            )
            extra.append(f"[相邻第 {neighbor_idx + 1} 页摘录，未经相关性验证]\n{excerpt}")
    return "\n".join(extra)


def get_retriever_details(question: str) -> list[dict[str, Any]]:
    """返回来源、页码、重排分数、实际送入模型的正文，便于调试和引用。"""
    if not question.strip():
        return []
    docs = get_vector_store().similarity_search(question, k=RAG_RECALL_K)
    if not docs:
        return []

    # 显式 Sigmoid，保证阈值比较在 0~1
    import torch
    scores = get_reranker().predict(
        [(question, doc.page_content) for doc in docs],
        activation_fn=torch.nn.Sigmoid(),
    )
    best: dict[str, tuple[float, Document]] = {}
    for doc, score_value in zip(docs, scores):
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            continue
        score = float(score_value)
        if RAG_RERANK_THRESHOLD is not None and score < RAG_RERANK_THRESHOLD:
            continue
        if parent_id not in best or score > best[parent_id][0]:
            best[parent_id] = score, doc
    if not best:
        return []

    client = _get_redis()
    parent_key = _parents_key()
    # 按分数排序并限制单来源页数，避免连续多页垄断
    selected: list[tuple[str, float, Document]] = []
    pages_per_source: dict[str, int] = {}
    for parent_id, (score, doc) in sorted(
        best.items(), key=lambda item: item[1][0], reverse=True
    ):
        source = str(doc.metadata.get("source", ""))
        if pages_per_source.get(source, 0) >= RAG_MAX_PAGES_PER_SOURCE:
            continue
        if client.hexists(parent_key, parent_id):
            selected.append((parent_id, score, doc))
            pages_per_source[source] = pages_per_source.get(source, 0) + 1
        if len(selected) >= RAG_TOP_N:
            break

    selected_ids = {parent_id for parent_id, _, _ in selected}
    remaining = RAG_MAX_CONTEXT_CHARS
    output: list[dict[str, Any]] = []
    for parent_id, score, doc in selected:
        page = client.hget(parent_key, parent_id)
        if not page:
            continue
        metadata = doc.metadata
        hit_start = max(0, int(metadata.get("start_index", 0)))
        page_index = int(metadata["page_index"])
        text = _trim_page(page, hit_start, RAG_PARENT_MAX_CHARS)
        neighbor = _neighbor_excerpt(
            client, parent_key, selected_ids, str(metadata["document_id"]),
            page_index, hit_start, len(doc.page_content), len(page)
        )
        if neighbor:
            text += "\n" + neighbor

        prefix = f"[来源：{metadata['filename']}｜第 {page_index + 1} 页｜重排分数：{score:.4f}]\n"
        available = remaining - len(prefix) - 2
        if available <= 0:
            break
        body = text[:available]
        output.append({
            "source": str(metadata["source"]),
            "filename": str(metadata["filename"]),
            "page": page_index + 1,
            "score": score,
            "parent_id": parent_id,
            "content": body,
            "context": prefix + body,
        })
        remaining -= len(prefix) + len(body) + 2
        if remaining <= 0:
            break
    return output


def get_retriever(question: str) -> str:
    """返回可直接交给大模型的带来源正文。"""
    return "\n\n".join(item["context"] for item in get_retriever_details(question))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 仅显式 RAG_REBUILD=1 时才全量删除
    if os.getenv("RAG_REBUILD", "0") == "1":
        del_rag(RAG_INDEX_NAME)

    files = [
        r"rag\9787502958572_L.pdf",
        r"rag\气象灾害防御_示范地区作物气象灾害防御指南_气象出版社.pdf",
        r"rag\气象科普_开远市气象科普手册_气象出版社.pdf",
        r"rag\国标GBT44709-2024_旅游景区雷电灾害防御技术规范.pdf",
        r"rag\国标GBT44954-2024_山岳地区雷电灾害防御技术规范.pdf",
        r"rag\国标GBT37926-2019_美丽乡村气象防灾减灾指南.pdf",
    ]
    for file in files:
        if Path(file).is_file():
            print(add_rag_pdf(file))
        else:
            log.warning("找不到 PDF，已跳过：%s", file)
    print(get_retriever("重大气象灾害之后要注意什么"))
