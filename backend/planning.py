"""确定性规划 / 调度纯函数（无 LLM / 网络 / Redis 依赖）。

负责任务规范化、DAG 校验、拓扑排序、语义键与调度决策。"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

ALLOWED_NODES = frozenset({"weather", "food", "travel"})
# COMPLETE_STATUSES：数据完整，可用于依赖门控与复用；DONE_STATUSES：已有终态（partial 不当完整数据）；
# SUCCESS_STATUSES：供生成回答阶段聚合使用（partial 需声明缺失）。
COMPLETE_STATUSES = frozenset({"success"})
DONE_STATUSES = frozenset({"success", "partial"})
SUCCESS_STATUSES = DONE_STATUSES
TERMINAL_STATUSES = frozenset({"success", "partial", "no_data", "failed"})
TRANSIENT_ERROR_CODES = frozenset({"timeout", "connection"})
TASK_ID_RE = re.compile(r"t\d+", re.IGNORECASE)

# 各类数据默认有效期（秒），超时结果不复用。
_DEFAULT_TTL = {"weather": 900, "travel": 600, "food": 86400, "default": 3600}

# ---------------------------------------------------------------- 基础工具


def normalize_text(text: Any) -> str:
    """归一化文本：去掉空白并转小写，用于任务语义比较。"""
    return re.sub(r"\s+", "", str(text or "")).lower()


def normalize_task(task: Any) -> dict:
    """把模型产出的任务对象规范化，避免类型漂移影响后续判定。"""
    task = task if isinstance(task, dict) else {}
    deps = task.get("task_depend") or []
    if isinstance(deps, (str, int)):
        deps = [deps]
    return {
        "task_id": str(task.get("task_id") or "").strip(),
        "task_name": str(task.get("task_name") or "").strip(),
        "task_question": str(task.get("task_question") or "").strip(),
        "task_node": str(task.get("task_node") or "").strip().lower(),
        "task_depend": [str(d).strip() for d in deps if str(d).strip()],
    }


def normalize_tasks(tasks: Any) -> list[dict]:
    if not isinstance(tasks, list):
        return []
    return [normalize_task(t) for t in tasks]


def fill_dependencies_from_text(tasks: list[dict]) -> None:
    """依据 task_question / task_name 中的 tN 引用补全依赖（仍需 validate_plan 校验）。"""
    task_ids = {str(t.get("task_id")) for t in tasks}
    for task in tasks:
        text = f'{task.get("task_question") or ""} {task.get("task_name") or ""}'
        refs = {m.lower() for m in TASK_ID_RE.findall(text)}
        refs = {r for r in refs if r in task_ids and r != str(task.get("task_id"))}
        if refs:
            task["task_depend"] = list(
                dict.fromkeys([*(task.get("task_depend") or []), *sorted(refs)])
            )


# ---------------------------------------------------------------- DAG 校验


def _cycle_errors(tasks: list[dict]) -> list[str]:
    order = _topological_ids(tasks)
    if len(order) == len(tasks):
        return []
    known = set(order)
    leftover = [str(t.get("task_id")) for t in tasks if str(t.get("task_id")) not in known]
    return [f"任务依赖存在循环或无法拓扑排序：{', '.join(leftover)}"]


def _topological_ids(tasks: list[dict]) -> list[str]:
    """Kahn 拓扑排序；存在环时返回已排出的前缀（长度 < 任务数）。"""
    by_id = {str(t.get("task_id")): t for t in tasks if t.get("task_id")}
    indegree = {tid: 0 for tid in by_id}
    children: dict[str, list[str]] = {tid: [] for tid in by_id}
    for tid, task in by_id.items():
        for dep in task.get("task_depend") or []:
            dep = str(dep)
            if dep in by_id and dep != tid:
                indegree[tid] += 1
                children[dep].append(tid)
    index = {str(t.get("task_id")): i for i, t in enumerate(tasks)}
    ready = sorted([tid for tid, n in indegree.items() if n == 0], key=lambda x: index.get(x, 0))
    order: list[str] = []
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for child in children.get(tid, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=lambda x: index.get(x, 0))
    return order


def topological_sort(tasks: list[dict]) -> list[dict]:
    """按依赖关系重排任务，保证"先驱在前"；存在环时原样返回。"""
    by_id = {str(t.get("task_id")): t for t in tasks}
    order = _topological_ids(tasks)
    if len(order) != len(tasks):
        return tasks
    return [by_id[tid] for tid in order]


def validate_plan(
    tasks: list[dict],
    *,
    max_tasks: int = 12,
    max_deps: int = 8,
    allowed_nodes: Iterable[str] = ALLOWED_NODES,
) -> list[str]:
    """返回结构化错误列表；空列表表示合法（错误文本回传 Planner 修复）。"""
    if not isinstance(tasks, list) or not tasks:
        return ["规划结果为空：未生成任何任务"]
    allowed = set(allowed_nodes)
    errors: list[str] = []
    if len(tasks) > max_tasks:
        errors.append(f"任务数量 {len(tasks)} 超过上限 {max_tasks}，请合并同源任务")

    ids: list[str] = []
    for index, raw in enumerate(tasks, 1):
        task = normalize_task(raw)
        tid = task["task_id"]
        if not tid:
            errors.append(f"第 {index} 个任务缺少 task_id")
            continue
        if tid in ids:
            errors.append(f"task_id 重复：{tid}")
        ids.append(tid)
        if task["task_node"] not in allowed:
            errors.append(
                f"任务 {tid} 的 task_node={task['task_node']!r} 非法，"
                f"只能是 {sorted(allowed)} 之一"
            )
        if not task["task_question"]:
            errors.append(f"任务 {tid} 缺少 task_question（子代理无记忆，必须自包含）")
        if len(task["task_depend"]) > max_deps:
            errors.append(f"任务 {tid} 的依赖数量超过上限 {max_deps}")

    id_set = set(ids)
    for task in tasks:
        tid = str(task.get("task_id") or "")
        for dep in task.get("task_depend") or []:
            dep = str(dep)
            if dep == tid:
                errors.append(f"任务 {tid} 存在自依赖")
            elif dep not in id_set:
                errors.append(f"任务 {tid} 依赖了不存在的任务 {dep}")
    errors.extend(_cycle_errors(tasks))
    return errors


# ---------------------------------------------------------------- 任务语义键


def compute_task_keys(tasks: list[dict]) -> dict[str, str]:
    """计算任务语义键（节点 + 提问 + 递归依赖键）；仅键一致的结果可复用。"""
    by_id = {str(t.get("task_id")): t for t in tasks if t.get("task_id")}
    memo: dict[str, str] = {}

    def key_of(tid: str, stack: tuple[str, ...] = ()) -> str:
        if tid in memo:
            return memo[tid]
        if tid in stack:  # 环由 validate_plan 拦截，此处防无限递归
            return "cycle"
        task = by_id.get(tid) or {}
        dep_keys = sorted(key_of(str(d), (*stack, tid)) for d in (task.get("task_depend") or []))
        raw = "\x1f".join(
            [
                str(task.get("task_node") or ""),
                normalize_text(task.get("task_question")),
                *dep_keys,
            ]
        )
        memo[tid] = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return memo[tid]

    for tid in by_id:
        key_of(tid)
    return memo


# ---------------------------------------------------------------- 结果有效性

def result_ttls() -> dict[str, int]:
    """按领域读取结果有效时间，允许通过环境变量覆盖。"""
    ttl = dict(_DEFAULT_TTL)
    for node in ("weather", "travel", "food", "default"):
        raw = os.getenv(f"RESULT_TTL_{node.upper()}", "").strip()
        if raw:
            try:
                ttl[node] = max(0, int(raw))
            except ValueError:
                pass
    return ttl


def result_is_fresh(
    entry: dict | None,
    *,
    now: float | None = None,
    ttl: dict[str, int] | None = None,
) -> bool:
    """结果是否仍在有效期内（实时天气/票务/航班不能无限期复用）。"""
    if not isinstance(entry, dict):
        return False
    try:
        updated_at = float(entry.get("updated_at") or 0)
    except (TypeError, ValueError):
        return False
    if updated_at <= 0:
        return False
    now = time.time() if now is None else now
    node = str(entry.get("task_node") or "")
    ttl = ttl or result_ttls()
    limit = ttl.get(node, ttl.get("default", 3600))
    return now - updated_at <= limit


def dep_signature(task: dict, effective_results: dict | None) -> list[list]:
    """前置结果版本指纹：依赖结果变化则下游旧结果失效。"""
    effective = effective_results or {}
    sig: list[list] = []
    for dep in task.get("task_depend") or []:
        info = effective.get(str(dep)) or {}
        if not isinstance(info, dict):
            info = {}
        sig.append([
            str(info.get("task_key") or ""),
            str(info.get("updated_at") or ""),
            int(info.get("attempt_count") or 0),
            str(info.get("status") or ""),
        ])
    return sig


def result_usable(
    entry: dict | None,
    key: str,
    *,
    invalidated: dict | None = None,
    now: float | None = None,
    ttl: dict[str, int] | None = None,
    allow_partial_missing: bool = False,
) -> bool:
    """结果是否可复用：状态、语义键、失效标记与时效均需满足。

    partial 有缺字段时仅允许同一 task_id 判定，不允许跨规划复用。"""
    if not isinstance(entry, dict):
        return False
    if entry.get("status") not in DONE_STATUSES:
        return False
    if str(entry.get("task_key") or "") != key:
        return False
    if entry.get("status") == "partial" and entry.get("missing_slots") and not allow_partial_missing:
        return False
    inval = (invalidated or {}).get(key)
    if inval is not None:
        try:
            if float(entry.get("updated_at") or 0) < float(inval):
                return False
        except (TypeError, ValueError):
            return False
    return result_is_fresh(entry, now=now, ttl=ttl)


def _dep_matches(entry: dict, task: dict, effective: dict) -> bool:
    """复用前校验依赖版本；旧格式（无 dep_sig）在有依赖时保守视为失效。"""
    stored = entry.get("dep_sig")
    expected = dep_signature(task, effective)
    if not isinstance(stored, list):
        return not expected
    return stored == expected


# ---------------------------------------------------------------- 调度决策

@dataclass
class SchedulePlan:
    total: int = 0
    ready: list[dict] = field(default_factory=list)
    completed: dict[str, dict] = field(default_factory=dict)  # 需要写入 all_task_results 的条目
    blocked: dict[str, list[str]] = field(default_factory=dict)  # 未完成且依赖不满足的任务
    all_done: bool = False
    keys: dict[str, str] = field(default_factory=dict)
    needs_supplement: dict[str, list[str]] = field(default_factory=dict)  # partial 缺字段的依赖


def plan_schedule(
    tasks: list[dict],
    results: dict | None,
    *,
    invalidated: dict | None = None,
    now: float | None = None,
    ttl: dict[str, int] | None = None,
) -> SchedulePlan:
    """计算本轮可派发 / 已完成 / 被阻塞的任务。

    已完成=同 id 且键一致、在有效期内；可复用=不同 id 但语义键一致。"""
    ordered = topological_sort(tasks)
    keys = compute_task_keys(ordered)
    results = results or {}
    now = time.time() if now is None else now
    ttl = ttl or result_ttls()

    reusable: dict[str, dict] = {}
    for res in results.values():
        if not isinstance(res, dict) or res.get("status") not in DONE_STATUSES:
            continue
        key = str(res.get("task_key") or "")
        if not key:
            continue
        if result_usable(res, key, invalidated=invalidated, now=now, ttl=ttl):
            reusable.setdefault(key, res)

    done: dict[str, dict] = {}
    effective: dict[str, dict] = dict(results)
    ready: list[dict] = []
    blocked: dict[str, list[str]] = {}
    needs_supplement: dict[str, list[str]] = {}

    for task in ordered:
        tid = str(task.get("task_id") or "")
        key = keys.get(tid, "")
        current = results.get(tid)
        usable = (
            isinstance(current, dict)
            and result_usable(
                current, key, invalidated=invalidated, now=now, ttl=ttl,
                allow_partial_missing=True,
            )
            and _dep_matches(current, task, effective)
        )
        if usable:
            done[tid] = current
            effective[tid] = current
            continue

        alternates = reusable.get(key)
        if alternates is not None and _dep_matches(alternates, task, effective):
            reused = {**alternates, "task_id": tid, "reused_from": str(alternates.get("task_id") or "")}
            done[tid] = reused
            effective[tid] = reused
            continue

        deps = [str(d) for d in (task.get("task_depend") or [])]
        missing = [d for d in deps if d not in done]
        if missing:
            blocked[tid] = missing
            lacking = [
                d for d in missing
                if ((results.get(d) or {}).get("status") == "partial")
                and (results.get(d) or {}).get("missing_slots")
            ]
            if lacking:
                needs_supplement[tid] = lacking
            continue
        lacking = [
            d for d in deps
            if str((done.get(d) or {}).get("status") or "") == "partial"
            and (done.get(d) or {}).get("missing_slots")
        ]
        if lacking:
            # 前置任务缺少下游必需字段：不允许提前执行
            blocked[tid] = lacking
            needs_supplement[tid] = lacking
            continue
        ready.append(task)

    completed = {
        tid: res for tid, res in done.items()
        if tid not in results or results.get(tid) != res
    }
    return SchedulePlan(
        total=len(ordered),
        ready=ready,
        completed=completed,
        blocked=blocked,
        all_done=len(done) == len(ordered),
        keys=keys,
        needs_supplement=needs_supplement,
    )


# ---------------------------------------------------------------- 结果分类


def group_step_results(step_results: dict | None) -> dict[str, dict]:
    """把 step_results 按 status 分组，容忍缺失字段。"""
    groups: dict[str, dict] = {}
    for tid, info in (step_results or {}).items():
        if not isinstance(info, dict):
            continue
        groups.setdefault(str(info.get("status") or "failed"), {})[str(tid)] = info
    return groups


def collect_numbers(value: Any, limit: int = 8) -> list[str]:
    """递归收集 key_data 中的数值（字符串化、去尾零），用于事实一致性抽检。"""
    found: list[str] = []

    def walk(item: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            text = f"{item:g}" if isinstance(item, float) else str(item)
            found.append(text)
        elif isinstance(item, dict):
            for v in item.values():
                walk(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                walk(v)

    walk(value)
    return found


def fact_consistency_hints(answer: str, results: dict | None) -> list[str]:
    """轻量事实抽检：回答应出现任务关键数值之一（仅作提示，不判失败）。"""
    text = str(answer or "")
    if not text.strip():
        return []
    hints: list[str] = []
    for tid, info in (results or {}).items():
        if not isinstance(info, dict) or info.get("status") not in SUCCESS_STATUSES:
            continue
        numbers = collect_numbers(info.get("key_data"))
        if not numbers:
            continue
        sample = numbers[:3]
        # 数字边界匹配，避免 25 命中 250
        matched = any(
            re.search(rf"(?<!\d){re.escape(n)}(?!\d)", text) for n in sample
        )
        if not matched:
            hints.append(
                f"任务 {tid} 的关键数值 {sample} 未出现在最终回答中，可能存在遗漏或数值不一致"
            )
    return hints


# ---------------------------------------------------------------- 规则路由

ROUTE_DIRECT = "direct"
ROUTE_SINGLE = "single"
ROUTE_MULTI = "multi"

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "weather": (
        "天气", "气温", "温度", "下雨", "降雨", "降水", "风力", "空气质量", "预报",
        "气象", "台风", "暴雨", "寒潮", "雷电", "紫外线", "湿度", "体感", "副热带",
        "季风", "锋面", "露点", "高温", "低温",
    ),
    "food": (
        "菜谱", "菜", "食谱", "做饭", "食材", "美食", "烹饪", "红烧", "汤", "早餐",
        "午餐", "晚餐", "营养", "热量", "好吃", "吃什么", "餐厅",
    ),
    "travel": (
        "高铁", "火车", "车次", "机票", "航班", "车站", "机场", "出行", "路线",
        "地图", "怎么走", "地铁", "打车", "12306", "车票", "余票", "行程", "换乘",
        "出发", "到达",
    ),
}

DIRECT_MARKERS = (
    "你好", "您好", "嗨", "hi", "hello", "谢谢", "多谢", "再见", "在吗", "早上好",
    "晚上好", "你是谁", "你能做什么", "你会什么", "讲个笑话", "聊聊天", "介绍一下你自己",
)

DEPENDENCY_MARKERS = (
    "结合", "根据", "基于", "在.{0,6}基础上", "先.{0,10}再", "然后", "接着", "同时",
    "顺便", "一并", "并推荐", "并规划", "之后", "再帮我",
)

EVALUATION_PATTERNS = (
    r"适合(去|去.{0,6})", r"适合.{0,10}(户外|活动|出游|出行|游览|玩)",
    r"值得(去|一去)", r"去.{0,6}(玩|旅游|游览)",
    r"一日游", r"行程(规划)?", r"路线(规划)?", r"值不值得", r"要不要去",
)

# 时间/日期类问题交给具备 get_time 工具的天气子 Agent，避免模型凭记忆作答
TIME_PATTERNS = (
    r"(现在|当前|此刻|目前).{0,4}(几点|时间|日期|几号)",
    r"今天(是)?(几号|周几|星期几)",
    r"北京时间",
    r"现在.{0,4}(多少点)",
)


def _match_domains(text: str) -> list[str]:
    return [domain for domain, words in DOMAIN_KEYWORDS.items() if any(w in text for w in words)]


def _is_direct(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in DIRECT_MARKERS)


def route_by_rules(question: str) -> tuple[str, str | None, str] | None:
    """规则优先路由；返回 None 表示需交给模型。

    多领域 / 依赖 / 评估类请求直接判 multi。"""
    text = str(question or "").strip()
    if not text:
        return (ROUTE_DIRECT, None, "空问题")
    if any(re.search(p, text) for p in DEPENDENCY_MARKERS):
        domains = _match_domains(text)
        return (ROUTE_MULTI, domains[0] if domains else None, "存在跨任务依赖表述")
    if any(re.search(p, text) for p in EVALUATION_PATTERNS):
        domains = _match_domains(text)
        return (ROUTE_MULTI, domains[0] if domains else None, "包含评估/规划类诉求")
    domains = _match_domains(text)
    if len(domains) >= 2:
        return (ROUTE_MULTI, None, "涉及多个领域")
    if any(re.search(p, text) for p in TIME_PATTERNS):
        return (ROUTE_SINGLE, "weather", "时间/日期查询")
    if len(domains) == 1:
        return (ROUTE_SINGLE, domains[0], "单一领域且无依赖表述")
    if _is_direct(text):
        return (ROUTE_DIRECT, None, "寒暄/自我介绍类")
    return None
