import asyncio
import json
import logging
import os
import re
from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, interrupt
from model import chat_model, chat_router_model
from subagent.food import get_food_graph
from subagent.travel import get_travel_graph
from subagent.weather import get_weather_graph
from typing_extensions import TypedDict
from typing import Literal
import operator
from bs4 import BeautifulSoup


def _step_results_reducer(current: dict, update: dict) -> dict:
    """step_results 的 reducer。

    普通 dict 按合并处理（并行步骤写回互不覆盖）；
    带 "__reset__" 标志时整值重置（新轮次开始时清空上轮残留，避免 reducer 合并导致无法清除）。
    """
    if isinstance(update, dict) and update.get("__reset__"):
        return dict(update.get("data") or {})
    return {**(current or {}), **(update or {})}


class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class Task(TypedDict):
    task_id : str
    task_name : str
    task_question : str
    task_node : Literal["weather", "food", "travel"]
    task_depend : Annotated[list[str],'依赖于哪几个任务的结果']

class plan_result(TypedDict):
    tasks : list[Task]

class feed_result(TypedDict):
    flag:bool
    info:str

class guard_result(TypedDict):
    flag:bool

class GraphState(InputState, OutputState):
    messages: Annotated[list, add_messages]
    tasks: list[Task]
    feedback_num: Annotated[int,operator.add]
    interrupt_num: Annotated[int,operator.add]
    error_num : Annotated[int,operator.add]
    step_results: Annotated[dict, _step_results_reducer] 
    all_task_results: Annotated[dict, _step_results_reducer]
    current_task: dict
    add_info : str
    feed_flag: bool
    feed_info:str



def clean_input(text) -> str:
    """基础输入过滤与文本归一化，在入口节点统一执行。"""
    if not text:
        return ""
    
    # 1. 剔除 HTML 标签及 script/style 等危险内容
    soup = BeautifulSoup(str(text), "html.parser")
    for tag in soup(["script", "style", "noscript", "head", "title"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    
    # 2. 全角转半角（保留常用中文标点）
    keep_punct = set("！＂＇（），：；？")
    text = "".join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E and c not in keep_punct
        else " " if c == "\u3000" else c
        for c in text
    )
    
    # 3. 剔除不可见控制字符（保留 \t, \n, \r）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # 4. 折叠连续空行与空白，并去除首尾空白
    text = re.sub(r"[^\S\n\r]+", " ", text)       # 行内连续空白折叠为单空格
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)  # 清除行首尾的空白字符
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3个及以上换行折叠为1个空行
    
    return text.strip()


def summarize_history(messages: list) -> str:
    """用模型把历史对话压缩成一段中文摘要。"""
    prompt = SystemMessage(
        content=(
            "你是一位对话历史摘要器。请把下面的历史对话压缩成一段简洁的中文摘要，"
            "保留所有对后续回答有用的信息：用户诉求、关键事实与已获得的数据"
            "（如时间、地点、数值、结论）。只输出摘要正文、不要任何前缀说明，全文控制在 200 个汉字以内。"
        )
    )
    resp = chat_model().invoke([prompt, *messages])
    return resp.content if hasattr(resp, "content") else str(resp)

def input_node(state: GraphState):
    """入口节点：清洗用户输入后写入；历史过长时先用模型摘要，最近 n 条强制保留。"""
    question = clean_input(state.get("question"))

    history = list(state.get("messages") or [])

    updates = {
        "messages": [HumanMessage(content=question)],
        "step_results": {"__reset__": True, "data": {}},
        "all_task_results": {"__reset__": True, "data": {}},
        'add_info': '',
        'feedback_num':-state.get("feedback_num",0),
        "error_num":-state.get("error_num",0),
        'interrupt_num':-state.get("interrupt_num",0),
        'error_info':'',
        'feed_info':''  
    }

    total_tokens = sum(len(m.content) for m in history) + len(question)
    if total_tokens // 4 > int(os.getenv("SUMMARY_THRESHOLD_TOKENS", "4096")):
        older = history[:-int(os.getenv("SUMMARY_KEEP_RECENT", '5'))]
        recent = history[-int(os.getenv("SUMMARY_KEEP_RECENT", '5')):]
        summary = summarize_history(older)
        removals = [RemoveMessage(id=m.id) for m in older if getattr(m, "id", None)]
        updates["messages"] = [
            SystemMessage(content=f"以下是历史对话摘要：\n{summary}"),
            *recent,  # 最近消息强制保留，不参与摘要
            *removals,
            HumanMessage(content=question),
        ]
    return updates

async def plan_node(state: GraphState):
    """规划节点：把问题拆解成一张"网络计划图"（每步 task/question/depends_on）。
    """
    prompt = SystemMessage(
        content=(
            "你是一位专业的任务规划师，把用户问题拆解成一张可执行的任务网络图（tasks 列表）。\n"
            "【字段】每个任务含 5 个字段："
            "task_id=任务编号，按 t1、t2、t3… 递增且不重复；"
            "task_name=一句话任务名（供前端展示）；"
            "task_question=抛给子代理的完整自包含提问，必须写全时间、地点等全部要素（子代理无记忆，不得写\"参考上文\"）；"
            "task_node=执行节点，只能是 weather（天气）/ food（美食）/ travel（出行）之一；"
            "task_depend=本任务依赖的任务编号列表，无依赖则填 []。\n"
            "【拆解规则】\n"
            "1. 粒度适中，不是越小越好：task_node 相同且 task_depend 一致的同源查询必须合并成一个任务"
            "（例如\"查询北京和郑州的明天天气\"合并为一个 weather 任务，提问里并列列出两地），不要拆成碎片化小任务；\n"
            "2. 宁串勿并：只有两个任务互不依赖对方结果时才允许并行；"
            "若一个任务必须先拿到另一任务的结果才能回答，必须把被依赖任务编号写入其 task_depend，形成串行链；\n"
            "3. 评估/判断类任务（如\"适合去 XX 吗\"\"建议穿什么\"\"值不值得去\"）必须依赖先行的查证类任务（天气/美食/路线），"
            "并在 task_question 中引用前置任务编号及其结果来提问，不得重复查询同类事实；\n"
            "4. 结合补充信息与已有结果：已成功完成或已有结果中的事实直接引用，避免重复已成功的任务、不必重新查询；\n"
            "5. 若问题完全不属于天气/美食/出行，且无法借助已有结果重组子任务，返回空 tasks 列表。\n"
            "【示例】问题\"明天北京天气如何？适合去颐和园吗？\"（串行依赖，而非并行）：\n"
            "- t1 task_node=weather task_depend=[] task_question=\"明天北京的天气如何？气温/晴雨/风力/空气质量。时间：明天，地点：北京\"\n"
            "- t2 task_node=travel task_depend=[\"t1\"] task_question=\"根据 t1 查到的明天北京天气，评估明天是否适合去颐和园游览并给建议。时间：明天，地点：北京颐和园\"\n"
            "【示例】问题\"北京和郑州明天天气如何？\"（合并为一个任务，而非拆成两个）：\n"
            "- t1 task_node=weather task_depend=[] task_question=\"查询北京和郑州两地的明天天气，两地各报气温/晴雨/风力/空气质量。时间：明天，地点：北京、郑州\"\n"
            "输出 tasks 必须按执行顺序排列：先列先驱任务、再列依赖它们的后续任务。"
        )
    )
    model = chat_model()
    model = model.with_structured_output(plan_result)
    result = await model.ainvoke([prompt, *state.get('messages')[:-1], HumanMessage(content=state.get("question")+ f'补充信息：{state.get("add_info","")}。现有结果:{state.get('all_task_results')}。错误信息:{state.get("error_info","")}。反馈意见:{state.get("feed_info","")}')])
    tasks = result.get("tasks", [])
    # 依据 task_question 中对其他任务 id 的显式引用（如"根据 t1""基于 t2"），
    # 自动补全漏填的 task_depend，确保"先查证、再评估"这类任务真正串行执行，
    # 而不被模型并行化（模型虽然会在提问里引用 t1，却经常不写 task_depend）。
    if tasks:
        task_ids = {str(t.get("task_id")) for t in tasks}
        for t in tasks:
            q = str(t.get("task_question") or "") + " " + str(t.get("task_name") or "")
            # 中文紧邻 t 时 \b 失效，故直接用 t\d+ 提取，再按 task_ids 集合过滤（只保留真实任务 id）
            refs = {m.lower() for m in re.findall(r"t\d+", q, flags=re.IGNORECASE)}
            refs = {r for r in refs if r in task_ids and r != str(t.get("task_id"))}
            if refs:
                t["task_depend"] = list(dict.fromkeys([*(t.get("task_depend") or []), *refs]))
    return {"tasks": tasks}


async def schedule_node(state: GraphState):
    """调度节点：根据 plan 生成的任务列表，按依赖关系派发可执行的子任务。
    """
    tasks = state.get("tasks", [])
    results = state.get("all_task_results", {})
    ready_tasks = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id in results.keys():
            continue  # 已完成的任务跳过
        depends_on = task.get("task_depend", [])
        if all(dep in results.keys() for dep in depends_on):
            ready_tasks.append(task)
    if ready_tasks:
        # Send 动态派发只携带显式 payload，不会自动继承父状态的 step_results / add_info；
        # 需把补充信息与本任务依赖的已成功结果一并下发，供 run_subagent 参考。
        add_info = state.get("add_info") or ""
        return [
            Command(goto=Send('run_subagent', {
                'current_task': task,
                'add_info': add_info,
                'deps_context': {
                    str(dep): (results.get(dep) or {})
                    for dep in (task.get("task_depend") or [])
                },
            })) for task in ready_tasks
        ]
    if results.keys():
        return Command(goto='merge', update={})
    else:
        return Command(goto='fallback', update={})


TASK_TO_GRAPH = {
    "weather": get_weather_graph,
    "food": get_food_graph,
    "travel": get_travel_graph,
}

async def run_subagent(state: GraphState) -> dict:
    """执行单个子代理步骤：带任务约束的指令调起子图，结果打包写入 step_results。

    无有效结果或调用异常时标记 status="failed"，供 check 判定后回 plan 重规划；
    串行步骤自动携带前置结果（deps_context）供子代理参考。
    节点经 schedule 的 Send 动态派发，不携带主图 config，子图按独立线程一次性执行。
    """
    step = state["current_task"]
    content, result, error = "", "", ""
    key_data = {}
    status = "failed"  # 默认失败；以下成功后以子图返回的 phase 覆盖
    try:
        prompt = (step.get("task_question") or "")
        deps = state.get("deps_context") or {}
        dep_parts = [
            f"- {k}: {v.get('result')}" for k, v in deps.items()
            if isinstance(v, dict) and v.get("result")
        ]
        if dep_parts:
            prompt += ("\n\n前置结果（已由其他子代理查证的事实，请直接引用，"
                       "不要再重复查询其中的数据）：\n" + "\n".join(dep_parts))
        add_info = state.get("add_info") or ""
        if add_info:
            prompt += f'\n\n补充信息：{add_info}'
        print(f"[DBG-run_subagent] add_info={state.get('add_info')!r} deps={list(deps)} prompt_tail={prompt[-120:]!r}", flush=True)
        out = await TASK_TO_GRAPH[step.get("task_node")]().ainvoke({"question": prompt})
        # 兼容子代理把内层结果嵌套在 result 字段的情况（result 可能是内层字典或其 JSON 文本）：
        # 统一把内层的 {result, key_data, status} 展开到顶层，确保 status="no_data" 能被 check
        # 正确识别并触发中断请教，同时使展示内容为干净的自然语言而非整段 JSON。
        if isinstance(out, dict):
            inner = out.get("result")
            if isinstance(inner, dict) and ("status" in inner or "key_data" in inner):
                out = {**{k: v for k, v in out.items() if k != "result"}, **inner}
            elif isinstance(inner, str) and inner.strip().startswith("{"):
                try:
                    parsed = json.loads(inner)
                except Exception:  # noqa: BLE001
                    parsed = None
                if isinstance(parsed, dict) and ("status" in parsed or "key_data" in parsed):
                    out = {**{k: v for k, v in out.items() if k != "result"}, **parsed}
        # 子代理返回 {result, key_data, status}；status 合法值为 success/no_data/failed
        sub_status = (out or {}).get("status") if isinstance(out, dict) else None
        content = (out or {}).get("result", "") if isinstance(out, dict) else out
        key_data = (out or {}).get("key_data", {}) if isinstance(out, dict) else {}
        if sub_status not in ("success", "no_data", "failed"):
            status = "success" if content else "failed"
        else:
            status = sub_status
        result = content
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        status = "failed"
    return {
        "step_results": {
            step["task_id"]: {
                "status": status,
                "key_data": key_data,
                "result": result,
                "error": error,
            }
        }
    }


async def check_node(state: GraphState) -> dict:
    """检查节点：汇总本轮所有步骤的执行结果，判断是否需要回 plan 重规划或继续派发。

    1. 若有失败步骤且未达重试上限，则回 plan 重规划；
    2. 若全部步骤完成，则汇总结果写入 all_task_results 并进入 merge 汇总；
    3. 若仍有就绪步骤，则回 schedule 续派（串行依赖）。
    """
    step_results = state.get("step_results", {})
    import collections
    results = collections.defaultdict(dict)
    for i, j in step_results.items():
        results[j.get("status")][i] = j
    no_data = results.get("no_data") or {}
    errors = results.get("failed") or results.get("error") or {}
    num = state.get("error_num", 0)
    num_interrupt = state.get("interrupt_num", 0)
    print(f"[DBG-check] statuses={dict((k,list(v.keys())) for k,v in results.items())} no_data={bool(no_data)} errors={bool(errors)} num={num} ni={num_interrupt}", flush=True)
    if no_data:
        if num_interrupt >= int(os.getenv("MAX_INTERRUPT", "3")):
            return Command(goto="fallback", update={"step_results": {"__reset__": True, "data": {}}, "all_task_results": results.get("success")})
        ask = no_data[next(iter(no_data))].get("result")
        add_info = state.get("add_info", "") + clean_input(interrupt(ask))
        return Command(goto="schedule", update={"step_results": {"__reset__": True, "data": {}}, "add_info": add_info, "all_task_results": results.get("success"), "interrupt_num": 1})
    if errors:
        if num >= int(os.getenv("MAX_ERROR", "3")):
            return Command(goto="fallback", update={"step_results": {"__reset__": True, "data": {}}, "all_task_results": results.get("success")})
        err_info = errors[next(iter(errors))].get("error")
        return Command(goto="plan", update={"step_results": {"__reset__": True, "data": {}}, "all_task_results": results.get("success"), "error_num": 1, "error_info": err_info})
    else:
        # 本批无 no_data/error：保留 step_results 供 schedule 判断"已完成"，用于续派串行依赖
        # 的后续步骤；全部完成时 schedule 自会进入 merge。
        # 不可在此清空 step_results——否则 schedule 会把已完成任务误判为待派发而无限循环。
        return Command(goto="schedule", update={"all_task_results": results.get("success")})

async def fallback_node(state: GraphState) -> dict:
    """兜底节点：当所有步骤失败或达到重试上限时，返回一个默认答复。"""
    return {"answer": "抱歉，这个问题我暂时无法回答。目前我只开放**天气**、**美食**与**出行**三类能力，您的问题似乎超出了我的服务范围。您可以试着问问我：某地的实时天气如何、某道菜的具体做法、或两地之间的车次怎么查询，我会尽我所能帮到您～"}


async def feedback_node(state: GraphState):
    prompt = [SystemMessage(
        "你是一位严谨的答案验收评估师。请结合\"参考数据\"核对给出的回答（最终汇总结果）是否满足全部条件：\n"
        "1. 切题：回答与用户原问题保持一致，未答非所问；\n"
        "2. 准确：与参考数据一致，没有虚构、矛盾或明显错误的数据（若参考数据本身冲突，以多数一致或注明来源的内容为准，不得判定为错误）；\n"
        "3. 完整：关键信息（时间、地点、数值、结论）均已给出（若参考数据为空，说明当前确实无数据可用，不得因此判定不完整）。\n"
        "全部满足时 flag 输出 true，否则 flag 输出 false，并在 info 中给出具体、可执行的修改意见"
        "（供重新规划时参考，要求简短、指向明确）。只需输出结构化字段，不要附加任何解释。"
    ), HumanMessage(content=f"回答：{state.get('answer')}\n\n参考数据：{state.get('all_task_results')}")]
    model = chat_model()
    model = model.with_structured_output(feed_result)
    result = await model.ainvoke(prompt)
    if result.get('flag'):
        return {'feed_flag':True,'feed_info':''}
    # 验收未通过：反馈给 plan 重规划，但设上限（默认 2），防止模型持续不通过时
    # 图在 feedback→plan 间无限循环，导致前端长时间停留在"验收未通过"上
    if state.get("feedback_num", 0) + 1 >= int(os.getenv("MAX_FEEDBACK", "2")):
        return {'feed_flag':True,'feed_info':''}
    return {'feed_flag':False,'feed_info':result.get('info'),'feedback_num':1}


def feed_path(state: GraphState):
    """反馈节点的分流路径：通过→guard→END；未通过→plan 重规划。"""
    if state.get("feed_flag"):
        return "guard"
    return "plan"

async def guard_node(state: GraphState):
    """输出防护节点：对所有最终答复做①内容安全审核②格式/协议检查。

    两项均通过则原样放行；任一未通过则替换为安全兜底文案并记录，避免
    有害内容外泄或非法格式流入下游。
    """
    prompt = [SystemMessage(
        "你是一位严格的内容安全与输出质量审核师。请审核待输出的回答是否同时满足以下条件：\n"
        "1. 内容安全：不包含色情、暴力、违法违规、歧视、诈骗、医疗/投资建议等风险不当或有害信息；\n"
        "2. 内容相关：答复与所问问题匹配，没有偏题、乱码或无意义的重复内容；\n"
        "3. 格式规范：没有明显的 HTML/脚本残留、脱敏失败或不可读的异常输出。\n"
        "仅当以上全部通过时 flag 输出 true，否则输出 false。只需输出结构化字段，不要附加任何解释。"
    ), HumanMessage(content=state.get('answer'))]
    model = chat_model()
    model = model.with_structured_output(guard_result)
    flag = await model.ainvoke(prompt)
    if flag.get('flag'):
        return {}
    return {'answer':'你的问题我无法回答'}

async def merge_node(state: GraphState):
    add_info = state.get("add_info") or ""
    sys = (
        "你是一位信息汇总师。请基于\"现有结果\"中的子任务结论，直接回答用户的原始问题，要求：\n"
        "1. 完整保留关键事实（时间、地点、数值、结论），不得虚构、篡改或自行补充数据；\n"
        "2. 若现有结果之间冲突，以多数一致或注明来源的内容为准并如实呈现，不得编造理由；\n"
        "3. 结构清晰、表达自然，可用 markdown 列表/表格增强可读性；\n"
        "4. 只基于现有结果与补充信息作答，不要声称做了额外查询；\n"
        "5. 若现有结果为空（所有子任务无数据或失败），如实说明未能获取相关信息，不要编造任何数据。"
    )
    context = f"补充信息：{add_info}\n现有结果：{state.get('all_task_results')}"
    prompt = [SystemMessage(content=sys), *state.get("messages"), HumanMessage(content=context)]
    model = chat_model()
    result = await model.ainvoke(prompt)
    # answer 必须是纯字符串（后续 feedback 用 HumanMessage(content=answer) 再次喂给模型）；
    # ainvoke 返回的是 AIMessage，需取其文本否则会被 pydantic 校验拒绝。
    return {'answer': result.content if hasattr(result, 'content') else str(result)}

_compiled_graph = None
async def get_agent_graph():
    """构建并缓存编译后的主图（首次调用时惰性初始化，之后直接复用）。

    主图使用异步 Redis 检查点（AsyncRedisSaver，需 await setup 绑定事件循环），
    以便同一 thread 内跨轮次保留对话历史（供摘要与上下文使用）。
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(GraphState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("input", input_node)
    graph.add_node("plan", plan_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("run_subagent", run_subagent)
    graph.add_node("check", check_node)
    graph.add_node("merge", merge_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("guard", guard_node)
    graph.add_edge(START, "input")
    graph.add_edge("input", "plan")
    # plan 产出任务后交给 schedule 派发；若 plan 为空（无可执行任务），
    # schedule 将直接进入 merge（由 merge 对空结果汇总兜底给出答复）
    graph.add_edge("plan", "schedule")
    # 步骤完成后静态汇合到 check（fan-in 只执行一次，能看到本批全部完成结果）
    graph.add_edge("run_subagent", "check")
    # check 后分流：失败→回 plan 重规划；全部完成→merge 汇总；仍有就绪步骤→回 schedule 续派（串行依赖）
    # 汇总后进入反馈验收；不通过且未达上限→回 plan 重规划，通过→进入输出防护 guard
    graph.add_edge("merge", "feedback")
    graph.add_conditional_edges(
        "feedback",
        feed_path,
        {"plan": "plan", "guard": "guard"},
    )
    # 兜底答复直接收口到 END（不经过 guard，文案为固定安全内容）；
    # 通过 guard 检测后的最终答复也在同一处收口到 END
    graph.add_edge("fallback", END)
    graph.add_edge("guard", END)

    saver = AsyncRedisSaver(
        os.getenv("REDIS_URL"),
        ttl={
            "default_ttl": 10080,  # 历史对话长期保留（7 天），之后才被回收
            "refresh_on_read": True,  # 读取时刷新 TTL（滑动续期）
        },
    )
    await saver.setup()
    _compiled_graph = graph.compile(checkpointer=saver)
    return _compiled_graph


# 文件级自测入口：仅直接运行 python agent.py 时执行；
# 若不加保护，uvicorn 导入本模块（api:app）时会重复触发一次真实 LLM 自测，
# 拖慢启动速度且可能在 Redis/模型不可用时导致整个 API 启动失败。
if __name__ == "__main__":
    agent_graph = asyncio.run(get_agent_graph())
    print(agent_graph.get_graph().print_ascii())