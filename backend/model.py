import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama, OllamaEmbeddings

# OpenCode Go 网关要求请求携带会话头与自有 User-Agent：
# - x-opencode-session：每个对话稳定的会话 id（用于路由与 prompt 缓存）
# - User-Agent：标识自有客户端，而非通用 SDK 名（否则可能被网关拒绝）
OPENCODE_HEADERS = {
    "x-opencode-session": os.getenv("opencode_session", "23124124"),
    "User-Agent": "wanxiang-agent/1.0",
}
@lru_cache(maxsize=1)
def chat_cloud_model():
    return init_chat_model(
        model=os.getenv("model", "deepseek-v4-flash"),
        model_provider=os.getenv("model_provider", "openai"),
        api_key=os.getenv("model_api"),
        base_url=os.getenv("base_url"),
        streaming=True,
        default_headers=OPENCODE_HEADERS,
        extra_body={"thinking": {"type": "disabled"}}
    )


@lru_cache(maxsize=1)
def chat_ollama_model():
    return ChatOllama(
        model=os.getenv("ollama_model", "qwen3.5:4b"),
        base_url=os.getenv("ollama_url", "http://localhost:11434"),
        reasoning=os.getenv("ollama_reasoning", "false"),
        temperature=os.getenv("ollama_temperature", "0.7"),
        num_ctx=8192,
    )

@lru_cache(maxsize=1)
def chat_model():
    if os.getenv("chat_model", "cloud").strip().lower() == "local":
        return chat_ollama_model()
    return chat_cloud_model()

@lru_cache(maxsize=1)
def chat_router_model():
    """路由/意图分类模型：由 .env 的 `route_model` 决定使用云端还是本地模型。

    - `cloud`（默认）：复用 OpenAI 兼容的 `chat_model()`
    - `local`：使用本地 Ollama 的 `chat_ollama_model()`
    """
    if os.getenv("route_model", "cloud").strip().lower() == "local":
        return chat_ollama_model()
    return chat_cloud_model()


@lru_cache(maxsize=1)
def embeddings_model():

    return OllamaEmbeddings(
        model=os.getenv("embeddings_model", "qwen3-embedding:latest"),
        base_url=os.getenv("ollama_url", "http://localhost:11434"),
        dimensions=1024
    )
