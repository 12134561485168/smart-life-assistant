import os
import uuid
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 网关要求会话头与自有 User-Agent：每进程生成随机会话前缀，按角色追加后缀；
# 可用环境变量 opencode_session 覆盖。
_PROCESS_SESSION = uuid.uuid4().hex


def _headers(role: str) -> dict:
    base = (os.getenv("opencode_session") or "").strip() or _PROCESS_SESSION
    return {
        "x-opencode-session": f"{base}:{role}",
        "User-Agent": "wanxiang-agent/1.0",
    }


def _cloud_model_name(role: str) -> str:
    """模型名按环境变量分角色解析：MODEL_<ROLE> > model。"""
    return os.getenv(f"MODEL_{role.upper()}") or os.getenv("model", "deepseek-v4-flash")


@lru_cache(maxsize=8)
def _cloud_model(role: str = "default"):
    return init_chat_model(
        model=_cloud_model_name(role),
        model_provider=os.getenv("model_provider", "openai"),
        api_key=os.getenv("model_api"),
        base_url=os.getenv("base_url"),
        streaming=True,
        default_headers=_headers(role),
        extra_body={"thinking": {"type": "disabled"}},
    )


@lru_cache(maxsize=1)
def chat_cloud_model():
    """默认角色的云端模型。"""
    return _cloud_model("default")


@lru_cache(maxsize=1)
def chat_ollama_model():
    return ChatOllama(
        model=os.getenv("ollama_model", "qwen3.5:4b"),
        base_url=os.getenv("ollama_url", "http://localhost:11434"),
        reasoning=os.getenv("ollama_reasoning", "false"),
        temperature=os.getenv("ollama_temperature", "0.7"),
        num_ctx=8192,
    )


def _resolve_model(role: str, channel_env: str = "chat_model"):
    """按角色解析模型通道：channel_env 决定 cloud / local。"""
    if os.getenv(channel_env, "cloud").strip().lower() == "local":
        return chat_ollama_model()
    return _cloud_model(role)


@lru_cache(maxsize=8)
def chat_model_for(role: str = "default"):
    return _resolve_model(role)


def chat_model():
    """默认对话模型。"""
    return chat_model_for("default")


def chat_router_model():
    """路由/意图分类模型：由 .env 的 `route_model` 决定云端或本地。"""
    return _resolve_model("router", "route_model")


def chat_planner_model():
    """规划模型：需要较强推理能力。"""
    return _resolve_model("planner")


def chat_subagent_model():
    """子 Agent 模型：需要稳定的工具调用能力。"""
    return _resolve_model("subagent")


def chat_merge_model():
    """汇总模型：优先响应速度。"""
    return _resolve_model("merge")


def chat_feedback_model():
    """反馈验收模型。"""
    return _resolve_model("feedback")


def chat_guard_model():
    """输出安全审核模型。"""
    return _resolve_model("guard")


@lru_cache(maxsize=1)
def embeddings_model():
    return OllamaEmbeddings(
        model=os.getenv("embeddings_model", "qwen3-embedding:latest"),
        base_url=os.getenv("ollama_url", "http://localhost:11434"),
        dimensions=1024,
        # 关闭 trust_env：避免全局代理把 localhost 请求也转发出去
        client_kwargs={"trust_env": False},
    )
