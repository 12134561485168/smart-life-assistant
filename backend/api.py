import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.load import load
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent import clean_input, get_agent_graph

# 显式加载 backend/.env，避免随启动目录读取错误配置
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

logger = logging.getLogger("uvicorn.error")

# 语义标记：线程最初的空检查点（用于重答首轮）
ROOT_MARKER = "__root__"

# 服务地址；实际监听由 uvicorn 启动参数决定
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8080"))
# 允许的跨域来源（逗号分隔）
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
# 问题 / 补充信息长度上限
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
MAX_RESUME_CHARS = int(os.getenv("MAX_RESUME_CHARS", "1000"))
# 主图单轮最大 superstep 数
RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "60"))
# 同一会话只允许一个生成任务，避免并发写检查点冲突
_active_threads: set[str] = set()
# 每用户每分钟请求上限（<=0 关闭），Redis 可用时跨进程生效
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_rate_windows: dict[str, list[float]] = {}

# ---------------- 服务端身份认证 ----------------
# 默认开启；AUTH_ENABLED=0 才退回用户名开发模式（勿用于生产）
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "1").strip() != "0"
TOKEN_TTL = int(os.getenv("TOKEN_TTL", "86400"))
AUTH_SECRET = (
    os.getenv("AUTH_SECRET")
    or os.getenv("APP_PASSWORD")
    or "wanxiang-dev-secret"
)


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign_token(username: str, ttl: int | None = None) -> tuple[str, int]:
    """签发无状态签名 Token（HMAC-SHA256），多进程共享同一密钥即可校验。"""
    exp = int(time.time()) + (TOKEN_TTL if ttl is None else ttl)
    payload = f"{username}|{exp}".encode()
    sig = hmac.new(AUTH_SECRET.encode(), payload, hashlib.sha256).hexdigest()[:40]
    return f"{_b64e(payload)}.{sig}", exp


def _verify_token(token: str) -> str | None:
    """校验签名与有效期，返回用户名；任何异常一律视为未认证。"""
    try:
        raw, sig = str(token or "").split(".", 1)
        payload = _b64d(raw).decode()
        username, exp_text = payload.rsplit("|", 1)
        expected = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:40]
        if not hmac.compare_digest(sig, expected):
            return None
        if int(exp_text) < time.time():
            return None
        if not username or len(username) > 64:
            return None
        return username
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


def _auth_user(request: Request | None) -> str:
    """从 Authorization: Bearer 解析身份；未认证一律 401。"""
    if not AUTH_ENABLED:
        return "anonymous"
    auth = (request.headers.get("authorization") if request is not None else "") or ""
    scheme, _, value = auth.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        username = _verify_token(value.strip())
        if username:
            return username
    raise HTTPException(401, "未登录或登录已过期")


def _check_rate_limit(username: str) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    now = time.time()
    if _r is not None:
        # Redis 计数：多进程/多 worker 部署下仍然生效
        key = f"wx:rate:{username}:{int(now // 60)}"
        try:
            count = int(_r.incr(key))
            if count == 1:
                _r.expire(key, 70)
            if count > RATE_LIMIT_PER_MINUTE:
                raise HTTPException(429, "请求过于频繁，请稍后再试")
            return
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("限流 Redis 读取失败，退回进程内限流：%s", exc)
    window = [t for t in _rate_windows.get(username, []) if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, "请求过于频繁，请稍后再试")
    window.append(now)
    _rate_windows[username] = window


class Question(BaseModel):
    # 空表示新会话，由后端生成会话密钥并回传
    thread_id: str | None = Field(
        default=None, max_length=64, description="会话标识（会话密钥）；空表示新会话"
    )
    question: str | None = Field(
        default=None, max_length=4000, description="用户问题；新提问时必填"
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="可选：从指定历史检查点分支继续（重新回答/选择续接点）；"
        f"传入 {ROOT_MARKER!r} 表示回到线程最初的空状态；"
        "缺省时普通续问自动使用该线程最新检查点（不会退回根检查点）",
    )
    resume: str | None = Field(
        default=None, max_length=2000,
        description="中断请教（interrupt）的恢复值：用户在澄清问询中补充的信息；"
        "非空时本次请求以 Command(resume=...) 恢复执行（check 节点拿到补充后回 schedule 续派剩余任务）",
    )
    username: str = Field(default="admin", max_length=64)


class Login(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class SessionRename(BaseModel):
    username: str = "admin"
    session_key: str
    title: str = ""  # 删除会话时不传标题，需默认值


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


def _owner_key(session_key: str) -> str:
    """会话归属标记：会话首次创建时绑定创建者，后续访问必须匹配。"""
    return f"wx:owner:{session_key}"


def _claim_owner(username: str, session_key: str) -> bool:
    """校验（或首次认领）会话归属，返回是否有权访问；Redis 不可用时直接放行。"""
    if _r is None:
        return True
    key = _owner_key(session_key)
    owner = _r.get(key)
    if owner is None:
        try:
            return bool(_r.set(key, username, nx=True))
        except Exception:  # noqa: BLE001
            return False
    return owner == username


def _assert_owner(username: str, session_key: str) -> None:
    if not _claim_owner(username, session_key):
        raise HTTPException(403, "无权访问该会话")


def _acquire_thread_lock(thread_id: str) -> tuple[str, str] | None:
    """获取会话执行锁，返回 (后端, 令牌)；已被占用返回 None。"""
    token = uuid.uuid4().hex
    if _r is not None:
        try:
            if _r.set(f"wx:lock:{thread_id}", token, nx=True, ex=900):
                return ("redis", token)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话锁 Redis 不可用，退回进程内锁：%s", exc)
    if thread_id in _active_threads:
        return None
    _active_threads.add(thread_id)
    return ("local", token)


def _release_thread_lock(thread_id: str, lock: tuple[str, str] | None) -> None:
    if not lock:
        return
    kind, token = lock
    if kind == "redis":
        try:
            # 原子 compare-and-delete，避免误删其他进程刚获取的锁
            _r.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] "
                "then return redis.call('del', KEYS[1]) else return 0 end",
                1, f"wx:lock:{thread_id}", token,
            )
        except Exception:  # noqa: BLE001
            pass
    else:
        _active_threads.discard(thread_id)


def _register_session(username: str, session_key: str, title: str):
    """登记会话；重复登记只更新时间，标题保留首次值。"""
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
    # 归属仅首次绑定，不可被他人接管
    try:
        _r.set(_owner_key(session_key), username, nx=True)
    except Exception:  # noqa: BLE001
        pass


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
    """检查点是否仍有效：存在非空状态即有效（线程 id 为 user_{session_key}）。"""
    try:
        st = await graph.aget_state({"configurable": {"thread_id": f"user_{session_key}"}})
        return bool(st and st.values)
    except Exception:  # noqa: BLE001
        # 线程不存在或访问异常一律视为失效
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
# 默认账号可用环境变量覆盖，生产必须显式配置
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "admin")


@app.post("/login")
def login(data: Login):
    """用户登录，成功后签发 HMAC 签名 Token（后续以 Bearer 传递）。"""
    if data.username == APP_USERNAME and data.password == APP_PASSWORD:
        token, exp = _sign_token(data.username)
        return {"ok": True, "username": data.username, "token": token, "expires_at": exp}
    raise HTTPException(401, "用户名或密码错误")


@app.get("/sessions")
async def sessions(request: Request):
    """当前用户的全部会话列表（按最近活动倒序），仅返回检查点仍有效的会话。"""
    username = _auth_user(request)
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
def session_rename(data: SessionRename, request: Request):
    """修改会话标题（必须登录且为会话归属者）。"""
    username = _auth_user(request)
    _assert_owner(username, data.session_key)
    existing = _r.hget(_sessions_key(username), data.session_key) if _r else None
    if existing is None:
        raise HTTPException(404, "会话不存在")
    try:
        obj = json.loads(existing)
    except (TypeError, json.JSONDecodeError):
        obj = {}
    obj["title"] = data.title.strip() or obj.get("title", "")
    obj["updated_at"] = int(time.time() * 1000)
    _r.hset(_sessions_key(username), data.session_key, json.dumps(obj, ensure_ascii=False))
    return {"ok": True}


@app.post("/sessions/delete")
def session_delete(data: SessionRename, request: Request):
    """删除会话：从该用户名下的会话表中移除，并清理其撤销标记。"""
    username = _auth_user(request)
    _assert_owner(username, data.session_key)
    if _r is not None:
        _r.hdel(_sessions_key(username), data.session_key)
        _r.delete(_revoked_key(username, data.session_key))
    return {"ok": True}


@app.post("/revoke")
def revoke(data: RevokeBody, request: Request):
    """把消息 id 标记为已撤销，/history 重建时过滤。"""
    username = _auth_user(request)
    if not data.thread_id.strip():
        raise HTTPException(400, "thread_id 不能为空")
    _assert_owner(username, data.thread_id.strip())
    key = _revoked_key(username, data.thread_id.strip())
    if data.message_ids and _r is not None:
        _r.sadd(key, *data.message_ids)
    return {"ok": True, "revoked": sorted(_revoked_ids(username, data.thread_id.strip()))}


# ============ 流式输出（SSE） ============
# 主图各节点 → 前端可折叠过程的标题与类型
NODE_LABELS = {
    "input": "开始处理",
    "router": "意图识别",
    "direct": "直接回答",
    "plan": "任务规划",
    "schedule": "步骤调度",
    "run_subagent": "子任务执行",
    "weather": "天气查询",
    "food": "美食菜谱",
    "travel": "出行规划",
    "check": "结果检查",
    "single_answer": "单任务回答",
    "merge": "结果汇总",
    "feedback": "反馈验收",
    "guard": "输出检测",
    "fallback": "兜底回复",
}


def _sse(event: str, data: dict) -> str:
    """把一条数据编码为 SSE 文本帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _plan_steps(tasks: list) -> list[dict]:
    """把 tasks 按执行波次分组（无依赖为第 1 波），返回 [{id, wave, question}]。"""
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
    # 按波次与编号排序，避免并行任务展示顺序漂移
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
    """把节点状态更新映射为 process 事件载荷（可能为空）。

    checkpoint_id 为该 superstep 落盘检查点，供前端绑定 AI 消息。"""
    if fields is None:
        fields = {}
    label = NODE_LABELS.get(node, node)
    task_map = task_map or {}
    if node == "schedule":
        return []  # 无有效进度，跳过
    if node == "run_subagent":
        # 每个 superstep 派发一批就绪子任务，结果在 step_results
        out = []
        for sid, info in (fields.get("step_results") or {}).items():
            info = info if isinstance(info, dict) else {}
            # 子任务标题用 task_question，缺失时退到节点标签
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
                "attempt_count": info.get("attempt_count"),
                "error_code": info.get("error_code"),
            })
        return out
    if node == "router":
        # 仅展示处理路径类型
        return [{
            "id": node, "node": node, "label": label, "type": "info",
            "checkpoint_id": checkpoint_id,
            "value": {"route": fields.get("route") or ""},
        }]
    if node == "plan":
        # 按波次展示计划步骤，不暴露内部节点类型
        return [{
            "id": node, "node": node, "label": label, "type": "plan",
            "checkpoint_id": checkpoint_id,
            "steps": _plan_steps(fields.get("tasks") or []),
            "version": fields.get("plan_version"),
            "errors": list(fields.get("plan_errors") or []),
        }]
    if node == "input":
        # 取清洗后的当前问题
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
        # 有 answer 字段即表示被拦截替换，否则放行
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
    # 其余节点：只透传安全的字段
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
    """把本轮过程内容按轮次锚写入 Redis（cid 为锚，供 /history 还原）。"""
    if _r is None or not cid or not collected:
        return
    try:
        _r.hset(_proc_key(thread_id), cid, json.dumps(collected, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("保存过程节点内容失败 thread=%s cid=%s: %s", thread_id, cid, exc)


@app.post("/answer/stream")
async def answer_stream(question: Question, request: Request):
    """以 SSE 流式返回主图执行过程与最终结果。

    每个 superstep 推送 `process`，结束推送 `done`；中断推送 `interrupt`；
    客户端断开后已落盘检查点仍保留。"""
    question_text = clean_input(question.question or "")
    resume_text = clean_input(question.resume or "")
    if not question_text and not resume_text:
        raise HTTPException(400, "question 不能为空")
    if len(question_text) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"问题过长，请控制在 {MAX_QUESTION_CHARS} 字以内")
    if len(resume_text) > MAX_RESUME_CHARS:
        raise HTTPException(400, f"补充信息过长，请控制在 {MAX_RESUME_CHARS} 字以内")

    # 身份取自服务端 Token，body username 仅开发模式有效
    if AUTH_ENABLED:
        username = _auth_user(request)
    else:
        username = (question.username or "admin").strip() or "admin"
    _check_rate_limit(username)
    session_key = (question.thread_id or "").strip()
    is_new = not session_key
    if is_new:
        # 首条消息：后端生成会话密钥并登记
        session_key = uuid.uuid4().hex
        title = question_text[:12] + ("…" if len(question_text) > 12 else "")
        _register_session(username, session_key, title)
    else:
        # 已有会话校验归属，防止仅凭 username 越权
        _assert_owner(username, session_key)

    graph = await get_agent_graph()
    thread_id = f"user_{session_key.strip()}"
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    # 显式 checkpoint_id 优先；缺省用最新检查点，不退回根检查点。
    marker = (question.checkpoint_id or "").strip()
    try:
        if marker:
            cid = await _resolve_cid(graph, thread_id, marker)
        else:
            states = await _history(graph, thread_id)
            cid = states[0].config["configurable"].get("checkpoint_id") if states else None
        if cid:
            cfg["configurable"]["checkpoint_id"] = cid
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("解析检查点失败 thread=%s", thread_id)
        raise HTTPException(500, f"解析检查点失败：{exc}") from exc

    # 并发执行会互相覆盖检查点，明确拒绝
    lock = _acquire_thread_lock(thread_id)
    if lock is None:
        raise HTTPException(409, "该会话正在生成回答，请等待当前回答结束后再发送")

    async def event_source():
        # 本轮过程事件，结束后按锚写入 Redis 供 /history 还原
        collected: list = []
        # 最后一次 process 事件的检查点，打断时作为落库锚
        last_process_cid: str | None = None
        # {task_id: task_question}，用于子任务步骤命名
        task_map: dict = {}
        try:
            # 建连后立即回传会话密钥，避免首问被打断时会话丢失
            yield _sse("session", {"session_key": session_key})
            # 有 resume 则恢复中断，否则按新问题启动
            stream_input = Command(resume=resume_text) if resume_text else {"question": question_text}
            async for chunk in graph.astream(stream_input, cfg, stream_mode="updates"):
                # check 节点 interrupt：推送澄清问题并结束本次响应
                if "__interrupt__" in chunk:
                    intr = chunk["__interrupt__"]
                    payload = intr[0] if isinstance(intr, tuple) and intr else intr
                    value = getattr(payload, "value", payload)
                    st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
                    cid_now = st.config["configurable"].get("checkpoint_id")
                    # 以中断检查点为锚落库过程内容
                    _save_proc(thread_id, cid_now, collected)
                    yield _sse("interrupt", {
                        "question": str(value or ""),
                        "checkpoint_id": cid_now,
                        # 首问即中断时一并回传会话密钥
                        "session_key": session_key,
                    })
                    return
                # 取当前检查点随事件下发，供打断时绑定 AI 消息
                st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
                cid_now = st.config["configurable"].get("checkpoint_id")
                for node, fields in chunk.items():
                    # 记录任务名供子任务步骤取标题
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
            # 本轮用户 / AI 消息 id，供前端撤销绑定
            msgs = st.values.get("messages") or []
            human_msg = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
            ai_msg = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            # 同时以"整轮结束"和"AI 写入前"两个锚写入，保证 /history 命中
            saved_cids = {after_cid}
            if ai_msg is not None:
                # /history 以 AI 写入前的检查点为锚取 proc，故归一到 before_cid
                before_cid = None
                async for st in graph.aget_state_history(
                    {"configurable": {"thread_id": thread_id}}, limit=200
                ):
                    msgs = (st.values or {}).get("messages") or []
                    if any(
                        getattr(_coerce_message(m), "id", None) == getattr(ai_msg, "id", None)
                        for m in msgs
                    ):
                        continue  # 仍含该 AI，继续向旧遍历
                    before_cid = st.config["configurable"].get("checkpoint_id")
                    break  # 已跨过该消息写入前的检查点
                if before_cid:
                    saved_cids.add(before_cid)
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
            # 客户端断开：以最后一次 process 的检查点为锚落库后退出
            _save_proc(thread_id, last_process_cid, collected)
            logger.info("客户端中断流式回答 thread=%s", thread_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式回答失败 thread=%s question=%s", thread_id, question_text)
            yield _sse("error", {"message": f"回答失败：{exc}"})
        finally:
            # 任何结束路径都释放会话锁
            _release_thread_lock(thread_id, lock)

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
async def history(thread_id: str, request: Request):
    """返回会话的可视消息历史与重答基线。

    已撤销消息会被过滤；每条 AI 消息带 re_answer_from（重答该轮应传的 checkpoint_id）。"""
    username = _auth_user(request)
    if not thread_id or not thread_id.strip():
        raise HTTPException(400, "thread_id 不能为空")
    thread_id = thread_id.strip()
    _assert_owner(username, thread_id)
    graph = await get_agent_graph()
    tid = f"user_{thread_id.strip()}"
    revoked = _revoked_ids(username, thread_id)
    states = await _history(graph, tid)
    if not states:
        return {"thread_id": thread_id, "messages": []}

    # 读取各轮过程内容：field=轮次锚，value=process 事件数组
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

    # 遍历全部检查点收集已完成轮（含各分支），供前端重建回答分支
    chrono = list(reversed(states))
    finished = []
    seen = set()
    for s in chrono:
        msgs = [_coerce_message(m) for m in (s.values.get("messages") or [])]
        if not msgs:
            continue
        # 只采集以 AI 结尾的轮；中途检查点会把上轮 AI 误挂到新问题
        if not isinstance(msgs[-1], AIMessage):
            continue
        ai = msgs[-1]
        human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if human is None:
            continue
        answer = (getattr(ai, "content", "") or "").strip()
        # 按消息 id 去重，避免重复提问或相同回答被错误合并
        human_id = getattr(human, "id", None)
        ai_id = getattr(ai, "id", None)
        if human_id or ai_id:
            key = (human_id, ai_id)
        else:
            # 旧检查点消息无 id 时退化为文本 + 检查点
            key = (human.content, answer, s.config["configurable"]["checkpoint_id"])
        if key in seen:
            continue
        seen.add(key)
        finished.append(
            {
                "q": human.content,
                "human_id": getattr(human, "id", None),
                "answer": answer,
                "cid": s.config["configurable"]["checkpoint_id"],
            }
        )

    out, last_cid, seen_q = [], None, set()
    for it in finished:
        if it["human_id"] in revoked:
            continue  # 该消息已被前端撤销，跳过不渲染
        # 同问题多回答：仅首次出现产出用户消息，其余为并列回答分支
        if it["q"] not in seen_q:
            out.append({"id": it["human_id"], "role": "human", "content": it["q"]})
            seen_q.add(it["q"])
        if not it["answer"]:
            continue  # 未产出答案，由下方 pending 逻辑补澄清
        out.append(
            {
                "id": None,
                "role": "ai",
                "content": it["answer"],
                "question": it["q"],  # 该轮提问，前端据此归并
                # 重答基线：上一轮落盘点（首轮为 ROOT_MARKER）
                "re_answer_from": last_cid or ROOT_MARKER,
                # 本轮回答落盘点：前端 AI 节点的 fork 锚点
                "checkpoint_id": it["cid"],
                # 该轮的思考过程
                "proc": _proc_for(it["cid"]),
            }
        )
        last_cid = it["cid"]
    # 末尾停在等待补充的用户消息时，追加一条 pending AI 澄清消息
    msgs_now = [_coerce_message(m) for m in (states[0].values.get("messages") or [])] if states else []
    if msgs_now and not isinstance(msgs_now[-1], AIMessage):
        intrs = getattr(states[0], "interrupts", None) or ()
        ask_text = str(getattr(intrs[0], "value", "") or "") if intrs else ""
        human_now = next((m for m in reversed(msgs_now) if isinstance(m, HumanMessage)), None)
        if ask_text and human_now is not None:
            q = getattr(human_now, "content", "") or ""
            cid_now = states[0].config["configurable"].get("checkpoint_id")
            if q and q not in seen_q and (getattr(human_now, "id", None) not in revoked):
                out.append({"id": getattr(human_now, "id", None), "role": "human", "content": q})
                seen_q.add(q)
            out.append(
                {
                    "id": None,
                    "role": "ai",
                    "content": ask_text,
                    "question": q,
                    # 重答基线：上一轮落盘点（首轮为 ROOT_MARKER）
                    "re_answer_from": last_cid or ROOT_MARKER,
                    # 前端据此绑定该 AI 消息
                    "checkpoint_id": cid_now,
                    "pending": True,  # 等待补充信息
                    # 中断前已发生的思考过程
                    "proc": _proc_for(cid_now),
                }
            )
    return {"thread_id": thread_id, "messages": out}


# 直接运行时以 uvicorn 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)