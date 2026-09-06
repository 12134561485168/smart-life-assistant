import asyncio
import json
import logging
import os
import time
import uuid

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.load import load
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent import clean_input, get_agent_graph

load_dotenv()  # 加载 .env 文件中的环境变量

logger = logging.getLogger("uvicorn.error")

# 语义标记：从线程最旧的初始（空）检查点继续，用于「重新回答/撤销第一轮」
ROOT_MARKER = "__root__"

# 后端服务端口（本文件仅定义；实际监听由 uvicorn 启动参数决定）
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8080"))
# 允许的跨域来源（逗号分隔），供前端 dev server 直接跨域调用
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")


class Question(BaseModel):
    # thread_id 为空串/缺失时视为"新会话"：后端在第一条用户消息时生成会话密钥并回传
    thread_id: str | None = Field(
        default=None, max_length=64, description="会话标识（会话密钥）；空表示新会话"
    )
    question: str | None = Field(
        default=None, description="用户问题；新提问时必填"
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="可选：从指定历史检查点分支继续（重新回答/选择续接点）；"
        f"传入 {ROOT_MARKER!r} 表示回到线程最初的空状态",
    )
    resume: str | None = Field(
        default=None,
        description="中断请教（interrupt）的恢复值：用户在澄清问询中补充的信息；"
        "非空时本次请求以 Command(resume=...) 恢复执行（check 节点拿到补充后回 plan 重规划）",
    )
    username: str = "admin"


class Login(BaseModel):
    username: str
    password: str


class SessionRename(BaseModel):
    username: str = "admin"
    session_key: str
    title: str = ""  # 删除会话时不传标题，需有默认值


class RevokeBody(BaseModel):
    username: str = "admin"
    thread_id: str  # 会话密钥
    message_ids: list[str]  # 被撤销的后端消息 id（含子树内所有消息）


# ---------- Redis（用户会话 / 撤销持久化，复用同一 Redis 库） ----------
try:
    _r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:26379"), decode_responses=True)
    _r.ping()
except Exception as exc:  # noqa: BLE001
    logger.warning("Redis 初始化失败，会话/撤销功能将不可用：%s", exc)
    _r = None


def _sessions_key(username: str) -> str:
    return f"wx:sessions:{username}"


def _revoked_key(username: str, session_key: str) -> str:
    return f"wx:revoked:{username}:{session_key}"


def _register_session(username: str, session_key: str, title: str):
    """把会话登记到该用户名下；重复登记只更新时间，标题保留首次生成值。

    会话表只保存会话 id（session_key）与展示所需的最小元数据（标题/时间），
    会话内容本身由 /history 按会话 id 从检查点重建，不在此冗余记录。
    """
    if _r is None:
        return
    now = int(time.time() * 1000)
    existing = _r.hget(_sessions_key(username), session_key)
    if existing:
        try:
            obj = json.loads(existing)
        except (TypeError, json.JSONDecodeError):
            obj = {}
        obj["updated_at"] = now
    else:
        obj = {
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
    _r.hset(_sessions_key(username), session_key, json.dumps(obj, ensure_ascii=False))


def _list_sessions(username: str) -> list[dict]:
    """按最近活动时间倒序返回该用户名下全部会话。"""
    if _r is None:
        return []
    out = []
    for k, v in ((_r.hgetall(_sessions_key(username)) or {})).items():
        try:
            obj = json.loads(v)
            obj["session_key"] = k
            out.append(obj)
        except (TypeError, json.JSONDecodeError):
            continue
    out.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
    return out


def _revoked_ids(username: str, session_key: str) -> set:
    if _r is None:
        return set()
    return set(_r.smembers(_revoked_key(username, session_key)) or [])


async def _session_alive(graph, session_key: str) -> bool:
    """会话对应的检查点是否仍有效：该线程存在非空状态即视为有效。

    会话表本身没有"失效"字段，但检查点可能被存储（TTL/清理）回收，
    失效后前端应不再展示。检查点线程 id 统一为 user_{session_key}。
    """
    try:
        st = await graph.aget_state({"configurable": {"thread_id": f"user_{session_key}"}})
        return bool(st and st.values)
    except Exception:  # noqa: BLE001
        # 线程不存在或访问异常一律视为失效，避免展示无法回放历史的会话
        return False


app = FastAPI(title="万象智能助手 API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _history(graph, thread_id: str) -> list:
    """线程内全部检查点（最新在前）。"""
    return [s async for s in graph.aget_state_history({"configurable": {"thread_id": thread_id}})]


def _coerce_message(m):
    """把检查点中以序列化 constructor dict 存储的消息还原为 Message 对象。"""
    if isinstance(m, dict) and m.get("lc") == 1 and m.get("type") == "constructor":
        return load(m, allowed_objects="messages")
    return m


async def _resolve_cid(graph, thread_id: str, checkpoint_id: str) -> str | None:
    """把请求中的 checkpoint_id 解析为线程内真实存在的检查点 id。

    - "__root__"：取最旧（初始空状态）检查点；线程无历史时返回 None
    - 其他值：必须真实存在于历史中，否则抛 400（防止误删 / 误回滚）
    """
    history = await _history(graph, thread_id)
    if not history:
        if checkpoint_id == ROOT_MARKER:
            return None
        raise HTTPException(400, "该会话还没有任何历史记录")
    if checkpoint_id == ROOT_MARKER:
        return history[-1].config["configurable"]["checkpoint_id"]
    existing = {s.config["configurable"]["checkpoint_id"] for s in history}
    if checkpoint_id not in existing:
        raise HTTPException(400, "checkpoint_id 不存在或已过期")
    return checkpoint_id


@app.get("/health")
def health():
    """健康检查：探活前端 -> 后端 -> Redis 链路是否可用。"""
    return {"status": "ok", "service": "万象智能助手 API"}


# ============ 用户登录 / 会话管理 ============
@app.post("/login")
def login(data: Login):
    """用户登录。目前仅允许账号 admin / 密码 admin（写死在前端及本处）。"""
    if data.username == "admin" and data.password == "admin":
        return {"ok": True, "username": data.username}
    raise HTTPException(401, "用户名或密码错误")


@app.get("/sessions")
async def sessions(username: str = "admin"):
    """当前用户的全部会话列表（按最近活动倒序），仅返回检查点仍有效的会话。

    会话表缓存不含"失效"字段，但检查点可能被存储回收，因此在返回前逐个检查
    对应线程是否仍有非空状态；失效会话不再展示给前端。
    """
    raw = _list_sessions(username)
    if not raw:
        return {"username": username, "sessions": []}
    graph = await get_agent_graph()
    alive = []
    for s in raw:
        if await _session_alive(graph, s["session_key"]):
            alive.append(s)
    return {"username": username, "sessions": alive}


@app.post("/sessions/rename")
def session_rename(data: SessionRename):
    """修改会话标题。"""
    existing = _r.hget(_sessions_key(data.username), data.session_key) if _r else None
    if existing is None:
        raise HTTPException(404, "会话不存在")
    try:
        obj = json.loads(existing)
    except (TypeError, json.JSONDecodeError):
        obj = {}
    obj["title"] = data.title.strip() or obj.get("title", "")
    obj["updated_at"] = int(time.time() * 1000)
    _r.hset(_sessions_key(data.username), data.session_key, json.dumps(obj, ensure_ascii=False))
    return {"ok": True}


@app.post("/sessions/delete")
def session_delete(data: SessionRename):
    """删除会话：从该用户名下的会话表中移除，并清理其撤销标记。

    删除后该会话不再出现在 /sessions 列表（即"只显示未失效的会话"）。
    会话对应的消息检查点另由检查点存储（RedisSaver）统一管理，不在此清理。"""
    if _r is not None:
        _r.hdel(_sessions_key(data.username), data.session_key)
        _r.delete(_revoked_key(data.username, data.session_key))
    return {"ok": True}


@app.post("/revoke")
def revoke(data: RevokeBody):
    """前端撤销消息时同步到后端：把这些消息 id 标记为已撤销，
    之后 /history 重建历史时会过滤掉它们，避免用户重新进入时错误渲染。"""
    if not data.thread_id.strip():
        raise HTTPException(400, "thread_id 不能为空")
    key = _revoked_key(data.username, data.thread_id.strip())
    if data.message_ids and _r is not None:
        _r.sadd(key, *data.message_ids)
    return {"ok": True, "revoked": sorted(_revoked_ids(data.username, data.thread_id.strip()))}


# ============ 流式输出（SSE） ============
# 主图各节点 → 前端可折叠过程的标题与类型
NODE_LABELS = {
    "input": "开始处理",
    "plan": "任务规划",
    "schedule": "步骤调度",
    "run_subagent": "子任务执行",
    "weather": "天气查询",
    "food": "美食菜谱",
    "travel": "出行规划",
    "check": "结果检查",
    "merge": "结果汇总",
    "feedback": "反馈验收",
    "guard": "输出检测",
    "fallback": "兜底回复",
}


def _sse(event: str, data: dict) -> str:
    """把一条数据编码为 SSE 文本帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _plan_steps(tasks: list) -> list[dict]:
    """把新 agent 的 tasks（含 task_depend 依赖）整理为按执行波次分组的计划步骤。

    无依赖的任务处于第 1 波；有依赖的任务波次 = 其依赖任务波次最大值 + 1。
    返回 [{id, wave, question}]，供前端按波次分组展示。
    """
    by_id = {str(t.get("task_id")): t for t in tasks if isinstance(t, dict) and t.get("task_id")}
    waves: dict = {}
    order: list = []

    def resolve(tid: str) -> int:
        if tid in waves:
            return waves[tid]
        t = by_id.get(tid)
        deps = t.get("task_depend") or [] if t else []
        w = 1
        for d in deps:
            if str(d) in by_id and str(d) != tid:
                w = max(w, resolve(str(d)) + 1)
        waves[tid] = w
        order.append(tid)
        return w

    steps = []
    for tid in by_id:
        resolve(tid)
    # 按波次、再按任务编号排序（编号即模型承诺的执行顺序），避免并行任务展示顺序漂移
    def _num(tid) -> int:
        digits = "".join(ch for ch in str(tid) if ch.isdigit())
        return int(digits) if digits else 0

    ordered = sorted(order, key=lambda tid: (waves[tid], _num(tid)))
    for tid in ordered:
        t = by_id[tid]
        steps.append({"id": tid, "wave": waves[tid], "question": (t.get("task_question") or "").strip()})
    return steps


def _node_to_events(node: str, fields, checkpoint_id: str | None = None,
                    task_map: dict | None = None) -> list[dict]:
    """把某个节点的一次状态更新映射为若干条 process 事件数据（可能为空列表）。

    返回的是结构化 dict（SSE 载荷），由调用方同时负责①推送前端、②写入 Redis
    （需求：流式输出时把对应节点内容落库，供 /history 重建历史会话的思考过程）。
    checkpoint_id 为该 superstep 完成后已落盘的当前检查点 id；
    前端在打断流式回答时，用它作为 AI 消息保存的 checkpoint（续问 / 重答的分支锚点）。
    """
    if fields is None:
        fields = {}
    label = NODE_LABELS.get(node, node)
    task_map = task_map or {}
    if node == "schedule":
        return []  # 无有效进度，跳过
    if node == "run_subagent":
        # 新 agent：run_subagent 一次性派发多个（可并行）子任务，写入 step_results
        out = []
        for sid, info in (fields.get("step_results") or {}).items():
            info = info if isinstance(info, dict) else {}
            # 子任务标题直接用规划出的任务提问（task_question）；task_map 万一缺失时退到
            # 节点标签即可，不再拼"子任务执行 · 步骤 tN"这类无意义后缀
            title = (task_map.get(str(sid)) or label).strip()
            out.append({
                "id": f"run_subagent:{sid}",
                "node": node,
                "label": label,
                "type": "tool",
                "checkpoint_id": checkpoint_id,
                "status": info.get("status"),
                "title": title,
                "content": info.get("result"),
                "key_data": info.get("key_data"),
                "error": info.get("error"),
            })
        return out
    if node == "plan":
        # 整理计划输出：按执行波次分组展示每步具体内容，不暴露内部 task/节点类型
        return [{
            "id": node, "node": node, "label": label, "type": "plan",
            "checkpoint_id": checkpoint_id, "steps": _plan_steps(fields.get("tasks") or []),
        }]
    if node == "input":
        # 展示输入处理结果：从该节点写入的 messages 中取出清洗后的当前问题
        processed = ""
        for m in fields.get("messages") or []:
            if isinstance(m, HumanMessage):
                processed = (m.content or "").strip()
        return [{
            "id": node, "node": node, "label": label, "type": "info",
            "checkpoint_id": checkpoint_id,
            "value": {"message": "输入处理：已接收并完成输入清洗与历史管理", "content": processed},
        }]
    if node == "merge":
        return [{
            "id": node, "node": node, "label": label, "type": "info",
            "checkpoint_id": checkpoint_id, "value": {"merged_answer": ""},
        }]
    if node == "feedback":
        passed = bool(fields.get("feed_flag"))
        value = {} if passed else {"feedback_text": (fields.get("feed_info") or "汇总结果未满足验收要求。")}
        return [{
            "id": node, "node": node, "label": label, "type": "info",
            "checkpoint_id": checkpoint_id, "value": value,
        }]
    if node == "guard":
        # 新 guard 未通过时会把 answer 替换为兜底文案；这里仅记录"已拦截/已放行"状态
        value = {"passed": not bool(fields.get("answer"))} if "answer" in fields else {"passed": True}
        return [{
            "id": node, "node": node, "label": label, "type": "info",
            "checkpoint_id": checkpoint_id, "value": value,
        }]
    if node == "fallback":
        return [{
            "id": node, "node": node, "label": label, "type": "info",
            "checkpoint_id": checkpoint_id, "value": {},
        }]
    # 其余节点（check 等）：只透传安全的标量/列表字段
    allowed = {"check": {"revision_reason", "retry_count", "check_retry_count",
                         "clarify_supplement"}}
    safe = {k: fields[k] for k in (allowed.get(node) or set()) if k in fields}
    return [{
        "id": node, "node": node, "label": label, "type": "info",
        "checkpoint_id": checkpoint_id, "value": safe,
    }]


def _proc_key(thread_id: str) -> str:
    """过程节点内容的 Redis key：wx:proc:{主图线程id}。"""
    return f"wx:proc:{thread_id}"


def _save_proc(thread_id: str, cid: str, collected: list) -> None:
    """把一轮流式回答收集到的过程节点内容写入 Redis（按轮次锚=结束检查点 id 分组）。

    cid 为该轮的锚：正常完成=回答落盘检查点；中断/打断=最后一次过程事件的检查点。
    前端 AI 节点的 checkpointId 与之对应，/history 据此还原各轮"思考过程"。
    """
    if _r is None or not cid or not collected:
        return
    try:
        _r.hset(_proc_key(thread_id), cid, json.dumps(collected, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("保存过程节点内容失败 thread=%s cid=%s: %s", thread_id, cid, exc)


@app.post("/answer/stream")
async def answer_stream(question: Question):
    """以 SSE 流式返回主图执行过程与最终结果（异步生成器）。

    每完成一个 superstep 就推送一条 `process` 事件（规划步骤、子代理/tools 结果、
    检查、汇总、反馈、输出检测等），并携带「当前已落盘检查点 id」——客户端打断时
    可据此把 AI 消息绑定到"当前"检查点，前端可折叠展示。
    全部结束后推送 `done` 事件，携带最终 `result` 与回答后检查点 id。

    特殊场景：
    - check 节点发现某步骤缺必要信息（status="no_data"）时暂停执行并向前端推送
      `interrupt` 事件（携带澄清问题与中断检查点）；前端展示问询、用户补充后，
      携带 checkpoint_id=中断点 + resume=补充信息再次请求，本接口以 Command(resume=...)
      恢复执行，check 拿到补充后回 plan 重新规划。
    - 客户端断开连接（前端点击「停止生成」）时，生成器收到 CancelledError 后退出，
      已完成的步骤检查点仍保留在 Redis，供后续续问 / 重答作为分支锚点。
    """
    question_text = clean_input(question.question or "")
    resume_text = (question.resume or "").strip() if isinstance(question.resume, str) else str(question.resume or "").strip()
    if not question_text and not resume_text:
        raise HTTPException(400, "question 不能为空")

    username = (question.username or "admin").strip()
    session_key = (question.thread_id or "").strip()
    is_new = not session_key
    if is_new:
        # 第一条用户消息：由后端随机生成会话密钥并登记到该用户名下
        session_key = uuid.uuid4().hex
        title = question_text[:12] + ("…" if len(question_text) > 12 else "")
        _register_session(username, session_key, title)

    graph = await get_agent_graph()
    thread_id = f"user_{session_key.strip()}"
    cfg = {"configurable": {"thread_id": thread_id}}
    # resume 与非 resume 均从 checkpoint_id 指向的检查点继续（中断恢复即 resume 场景）
    marker = question.checkpoint_id or ROOT_MARKER

    try:
        cid = await _resolve_cid(graph, thread_id, marker)
        if cid:
            cfg["configurable"]["checkpoint_id"] = cid
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("解析检查点失败 thread=%s", thread_id)
        raise HTTPException(500, f"解析检查点失败：{exc}") from exc

    async def event_source():
        # 本轮流式回答收集到的过程节点内容（dict 载荷），结束后按轮次锚写入 Redis，
        # 供 /history 重建历史会话时还原各轮"思考过程"
        collected: list = []
        # 最后一次实际推送 process 事件时的检查点 id（打断时作为该轮的落库锚点）
        last_process_cid: str | None = None
        # 本轮 plan 产生的 {task_id: task_question}，用于给 run_subagent 的子任务步骤命名
        task_map: dict = {}
        try:
            # 连接建立后立即回传本请求对应的会话密钥：新会话首问即使随即被
            # 客户端打断（停止生成，连接断开后无法再回传），前端也已拿到 key，
            # 会话不会因打断而丢失；已有会话幂等重设，无副作用
            yield _sse("session", {"session_key": session_key})
            # 恢复中断时用 Command(resume=...) 继续；否则按新问题启动
            stream_input = Command(resume=resume_text) if resume_text else {"question": question_text}
            async for chunk in graph.astream(stream_input, cfg, stream_mode="updates"):
                # 图在 check 节点 interrupt 暂停：向前端推送澄清问题后结束本次响应
                if "__interrupt__" in chunk:
                    intr = chunk["__interrupt__"]
                    payload = intr[0] if isinstance(intr, tuple) and intr else intr
                    value = getattr(payload, "value", payload)
                    st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
                    cid_now = st.config["configurable"].get("checkpoint_id")
                    # 中断也是该轮的自然结尾：以中断检查点为锚把已收集的过程内容落库
                    _save_proc(thread_id, cid_now, collected)
                    yield _sse("interrupt", {
                        "question": str(value or ""),
                        "checkpoint_id": cid_now,
                        # 新会话首问即被中断时，把生成的会话密钥一并回传，供前端关联该会话
                        "session_key": session_key,
                    })
                    return
                # 该 superstep 完成且已落盘，取当前检查点随事件下发；
                # 打断时前端据此把 AI 消息绑定到"当前"检查点
                st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
                cid_now = st.config["configurable"].get("checkpoint_id")
                for node, fields in chunk.items():
                    # 记录本轮规划的任务名，供 run_subagent 子任务步骤取标题
                    if node == "plan":
                        for t in (fields or {}).get("tasks") or []:
                            if isinstance(t, dict) and t.get("task_id"):
                                task_map[str(t["task_id"])] = (t.get("task_question") or "").strip()
                    for ev in _node_to_events(node, fields, cid_now, task_map):
                        collected.append(ev)
                        if ev.get("checkpoint_id"):
                            last_process_cid = ev.get("checkpoint_id")
                        yield _sse("process", ev)
            st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            after_cid = st.config["configurable"].get("checkpoint_id")
            result = st.values.get("answer", "") if st.values else ""
            # 本轮新增的 HumanMessage/AIMessage 的 langchain id，供前端撤销持久化绑定
            msgs = st.values.get("messages") or []
            human_msg = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
            ai_msg = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            # 正常完成：把本轮全部过程节点内容写入 Redis。
            # 前端展示"思考过程"时以该轮 AI 消息的落盘点为锚，而 after_cid 是
            # 整个 superstep（含 guard 等无消息节点）结束后的检查点，二者通常不同；
            # 故以该轮 AI 消息对应的检查点为锚再写一份，保证 /history 能命中 proc。
            saved_cids = {after_cid}
            if ai_msg is not None:
                # /history 里每条 AI 消息以"消息首次写入的检查点"为锚取 proc，
                # 该锚位于 after_cid（superstep 全部结束后）之前；从最新检查点
                # 向旧遍历，直到遇到不再包含该消息的检查点为止，即得到该写入锚。
                anchor = None
                async for st in graph.aget_state_history(
                    {"configurable": {"thread_id": thread_id}}, limit=200
                ):
                    msgs = (st.values or {}).get("messages") or []
                    if any(
                        getattr(_coerce_message(m), "id", None) == getattr(ai_msg, "id", None)
                        for m in msgs
                    ):
                        anchor = st.config["configurable"].get("checkpoint_id")
                    else:
                        break  # 已跨过该消息首次写入前的检查点
                if anchor:
                    saved_cids.add(anchor)
            for anchor_cid in saved_cids:
                _save_proc(thread_id, anchor_cid, collected)
            yield _sse("done", {
                "result": result,
                "session_key": session_key,
                "human_message_id": getattr(human_msg, "id", None),
                "ai_message_id": getattr(ai_msg, "id", None),
                "after_checkpoint_id": after_cid,
            })
        except asyncio.CancelledError:
            # 客户端打断（连接断开）：图已在每个 superstep 落盘检查点，
            # "当前 checkpoint"就是最后一次 process 事件携带的那个，以它为锚把
            # 已收集的过程内容落库（与前端保存 AI 消息用的 checkpoint 一致），记录后退出
            _save_proc(thread_id, last_process_cid, collected)
            logger.info("客户端中断流式回答 thread=%s", thread_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式回答失败 thread=%s question=%s", thread_id, question_text)
            yield _sse("error", {"message": f"回答失败：{exc}"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/history")
async def history(thread_id: str, username: str = "admin"):
    """获取指定会话的可视消息历史与“重答基线”。

    每条人工消息附带其后端消息 id（供撤销持久化）；
    已通过 /revoke 标记为撤销的消息会被过滤，避免重新进入时错误渲染。
    每条 AI 消息附带 re_answer_from：重答该轮时 /answer/stream 应传入的 checkpoint_id
    （线程首轮为 "__root__"，其余轮为上一轮回答后的检查点）。
    """
    if not thread_id or not thread_id.strip():
        raise HTTPException(400, "thread_id 不能为空")
    thread_id = thread_id.strip()
    graph = await get_agent_graph()
    tid = f"user_{thread_id.strip()}"
    revoked = _revoked_ids(username, thread_id)
    states = await _history(graph, tid)
    if not states:
        return {"thread_id": thread_id, "messages": []}

    # 读取本会话各轮的"过程节点内容"（/answer/stream 流式回答时写入，
    # hash 的 field=轮次锚(该轮结束检查点 id)，value=该轮所有 process 事件 dict 的 JSON 数组）
    proc_map = {}
    if _r is not None:
        try:
            proc_map = _r.hgetall(_proc_key(tid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取过程节点内容失败 thread=%s: %s", tid, exc)

    def _proc_for(cid: str | None) -> list:
        if not cid or not proc_map.get(cid):
            return []
        try:
            return json.loads(proc_map[cid])
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析过程节点内容失败 thread=%s cid=%s: %s", tid, cid, exc)
            return []

    latest_values = states[0].values
    # 新 agent 不把 AI 回答写入 messages，历史仅含用户消息与（可能的）摘要；
    # 每轮最终答案保存在该轮结束检查点的 answer 字段中，需据此重建"用户→AI"对。
    dialogue = [
        _coerce_message(m)
        for m in latest_values.get("messages", [])
        if isinstance(_coerce_message(m), HumanMessage)
    ]

    # 记录"每个用户消息所在轮的结束检查点 id"：从最旧到最新遍历，覆盖写入，
    # 最终保留的是该用户消息最后一次作为消息末尾的那个检查点（= 该轮回答落盘位置）
    chrono = list(reversed(states))
    cid_values: dict = {}      # checkpoint_id -> 该检查点的 state 值
    round_end: dict = {}       # 用户消息 id -> 该轮结束检查点 id
    for s in chrono:
        cid = s.config["configurable"]["checkpoint_id"]
        cid_values[cid] = s.values
        msgs = s.values.get("messages") or []
        tail = None
        for m in reversed(msgs):
            m = _coerce_message(m)
            if isinstance(m, HumanMessage):
                tail = m
                break
        mid = getattr(tail, "id", None) if tail is not None else None
        if mid:
            round_end[mid] = cid

    def _answer_at(cid):
        v = cid_values.get(cid)
        return ((v or {}).get("answer") or "").strip()

    out, prev_after, last_human = [], None, None
    for m in dialogue:
        mid = getattr(m, "id", None)
        if mid in revoked:
            continue  # 该消息已被前端撤销，跳过不渲染
        last_human = m.content
        out.append({"id": mid, "role": "human", "content": m.content})
        after = round_end.get(mid)
        answer = _answer_at(after)
        if not answer:
            continue  # 本轮尚未产出答案（中断等待补充/被打断），由下方 pending 逻辑补齐
        out.append(
            {
                "id": None,
                "role": "ai",
                "content": answer,
                "question": last_human,  # 该轮提问（重答时直接复用）
                # 重答基线：上一轮落盘点（首轮为 ROOT_MARKER）
                "re_answer_from": prev_after or ROOT_MARKER,
                # 本轮回答落盘点：前端树模型中 AI 节点的 fork 锚点
                "checkpoint_id": after,
                # 该轮的"思考过程"（流式回答时落库，按轮次锚=回答落盘点取回）
                "proc": _proc_for(after),
            }
        )
        prev_after = after
    # 图在 check 节点 interrupt 暂停（等待用户补充信息）：AI 的澄清问题此时只存在于
    # 检查点的 interrupts 中，未写入 messages，本轮也没有 answer。
    # 若对话末尾是用户消息且存在待处理的澄清提问，则把它作为一条"等待补充"的 AI 消息追加。
    if out and out[-1].get("role") != "ai":
        intrs = getattr(states[0], "interrupts", None) or ()
        if intrs:
            ask_text = str(getattr(intrs[0], "value", "") or "")
            if ask_text:
                cid_now = states[0].config["configurable"].get("checkpoint_id")
                out.append(
                    {
                        "id": None,
                        "role": "ai",
                        "content": ask_text,
                        "question": last_human,
                        # 重答基线：上一轮落盘点（首轮为 ROOT_MARKER）
                        "re_answer_from": prev_after or ROOT_MARKER,
                        # 中断检查点：前端据此把该 AI 消息绑定到"暂停等待补充"的锚点
                        "checkpoint_id": cid_now,
                        "pending": True,  # 标记为等待补充信息
                        # 该轮已发生的"思考过程"（直到 check 中断点为止），供前端还原
                        "proc": _proc_for(cid_now),
                    }
                )
    return {"thread_id": thread_id, "messages": out}


# 直接运行 python api.py 时以 uvicorn 启动服务；调试流式接口请直接访问 /answer/stream。
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)