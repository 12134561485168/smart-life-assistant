"""子 Agent 共用的 MCP 工具缓存与工具证据采集。

缓存带 asyncio.Lock 并发保护，并支持显式失效。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Awaitable

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

log = logging.getLogger(__name__)

_MCP_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp.json")
# 单个 MCP 工具调用上限；远端工具挂起时及时失败，避免拖垮整个子 Agent
MCP_TOOL_TIMEOUT = float(os.getenv("MCP_TOOL_TIMEOUT", "45"))


def _with_timeout(tool: Any, seconds: float = MCP_TOOL_TIMEOUT) -> Any:
    """给工具协程加超时，超时抛出可读的 TimeoutError。"""
    coro = getattr(tool, "coroutine", None)
    if coro is None or seconds <= 0:
        return tool

    async def _timed(*args: Any, **kwargs: Any):
        try:
            return await asyncio.wait_for(coro(*args, **kwargs), timeout=seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"工具 {tool.name} 调用超时（>{seconds:g}s）") from exc

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_timed,
        response_format=tool.response_format,
    )


def load_mcp_section(section: str) -> dict:
    """读取 mcp.json 中指定子 Agent 的连接配置。"""
    with open(_MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)[section]


async def load_tools_resilient(client: Any, connections: dict) -> list:
    """逐个 MCP 服务加载工具，单个服务故障不影响其余服务；全部失败才抛错。"""
    results = await asyncio.gather(
        *(client.get_tools(server_name=name) for name in connections),
        return_exceptions=True,
    )
    tools: list = []
    errors: list[str] = []
    for name, result in zip(connections, results):
        if isinstance(result, BaseException):
            root: BaseException = result
            while isinstance(root, BaseExceptionGroup) and root.exceptions:
                root = root.exceptions[0]
            errors.append(f"{name}: {root}")
            log.warning("MCP 服务 %s 工具加载失败，已跳过：%s", name, root)
            continue
        tools.extend(_with_timeout(tool) for tool in result)
    if not tools:
        raise RuntimeError("MCP 工具全部不可用：" + "; ".join(errors))
    return tools


class ToolsCache:
    """MCP 工具列表缓存：并发安全的首次加载 + 失败后可失效重连。"""

    def __init__(self) -> None:
        self._tools: list | None = None
        self._lock = asyncio.Lock()

    async def get(self, loader: Callable[[], Awaitable[list]]) -> list:
        if self._tools is not None:
            return self._tools
        async with self._lock:
            if self._tools is None:
                self._tools = await loader()
        return self._tools

    def invalidate(self) -> None:
        """MCP 会话异常时清空缓存，下次调用重新握手。"""
        self._tools = None


def collect_tool_evidence(messages: list | None, limit: int = 8, max_chars: int = 600) -> list[dict]:
    """从 ToolMessage 提取真实工具响应片段作为证据。"""
    evidence: list[dict] = []
    for message in messages or []:
        is_tool = isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool"
        if not is_tool:
            continue
        raw: Any = getattr(message, "content", "")
        if isinstance(raw, list):
            content = " ".join(str(item) for item in raw)
        else:
            content = str(raw)
        entry = {
            "tool": getattr(message, "name", None) or getattr(message, "tool_name", None) or "tool",
            "content": content[:max_chars],
        }
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        evidence.append(entry)
        if len(evidence) >= limit:
            break
    return evidence
