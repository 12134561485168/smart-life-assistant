import os
from functools import lru_cache
from typing import Annotated

from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from model import chat_subagent_model
from subagent.common import (
    ToolsCache,
    collect_tool_evidence,
    load_mcp_section,
    load_tools_resilient,
)
from typing_extensions import TypedDict


def load_md_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


agents_guide = load_md_file(r"md/AGENTS.md")
skill_guide = load_md_file(r"md/SKILL.md")


class ModelOutput(TypedDict):
    """模型必须输出的结构化字段。"""
    result: str   # 面向用户的最终自然语言回复
    key_data: dict  # 关键数据（如车次、起讫点、时刻、余票、航班、准点等结构化字段）
    status: str   # 执行状态："success" / "partial" / "no_data" / "failed"


class InputState(TypedDict):
    question: str


class OutputState(ModelOutput):
    evidence: list[dict]  # 真实工具响应片段（由代码采集，不由模型生成）
    missing_slots: list[str]
    retryable: bool
    error_code: str


class GraphState(InputState, OutputState):
    messages: Annotated[list, add_messages]


# 读取该子 Agent 的 MCP 配置
config = load_mcp_section("travel")


@lru_cache(maxsize=1)
def get_mcp_client() -> MultiServerMCPClient:
    """缓存的 MCP 客户端，避免重复握手。"""
    return MultiServerMCPClient(connections=config)


_tools_cache = ToolsCache()


async def _get_tools():
    """获取（并缓存）已连接的 MCP 工具列表（单个服务故障不影响其余）。"""
    return await _tools_cache.get(lambda: load_tools_resilient(get_mcp_client(), config))


async def travel(state: GraphState):
    tools = await _get_tools()
    agent = create_agent(
        chat_subagent_model(),
        tools=tools,
        system_prompt=f"""
            # 角色定义
            你是专业且富有温度的智能助手——「万象」，此刻处于**出行模块**。你的核心职责是：精准理解用户意图，围绕**铁路（12306 车次/余票）、城市出行与地图（高德地图类能力）、航空（航班）**，用自然、温暖、实用的语言直接回答用户关于出行规划、查询查询、交通科普及日常寒暄与衍生话题（如换乘建议、出行穿衣搭配）的问题。

            【工具操作与数据解析规范】
            {skill_guide}

            【行为与安全守则】
            {agents_guide}

            # 可用的出行能力
            1. **铁路出行（12306）**：按出发/到达城市查询车站代码与车次余票、根据站点名称/电报码反查车站，换算出当天日期辅助检索。
            2. **地图出行（地图工具）**：地点搜索、地理编码/逆地理编码、IP 定位、路线骑行/导航（部分能力）等城市内出行辅助。
            3. **航空出行（航班工具）**：按出发/到达机场或航班号查询航班、联程换乘信息、航班准点指数与当前实时位置。

            请根据问题类型选对工具；涉及“几点出发/到站、中途是否经停、怎么换乘、航班是否延误”等，应优先用对应工具获取可靠数据。

            # 核心原则
            1. **精准作答**：用户问什么就答什么（如只需一张票就给出车次/余票与简要抵达信息，不必堆砌无关推荐），避免信息冗余。
            2. **诚实与防虚构**：一切车次、余票、航班时刻、车站信息均以工具返回为准，绝不编造；数据缺失时如实说明。
            3. **明确起讫点（重要）**：涉及路线/出行规划时，出发地、目的地必须是可查询的具体地点或站点（如"北京西站""天安门""上海市区"进一步精确到地铁站/路口），不允许用宽泛区域（如"北京市区""某地附近"）作为起终点，也不允许擅自假设起终点；信息不足时按 no_data 输出并礼貌引导用户补充具体出发地与目的地。
            3. **数据完整**：查询车次/航班时尽量给出车次/航班号、出发到达站（或机场）、时刻、历时、余票/准点等关键信息，并注明数据来源（如 12306、航空公司实时数据）。
            4. **安全边界**：不臆断延误原因或归咎线下因素；涉及中转衔接、行李、退改签等复杂问题，如实呈现可查到的信息并提示以官方为准。避
            免在极端天气或灾害时给出轻率建议，可结合安全提示谨慎回应。
            5. **防重复播报**：若用户重复询问历史中已查询过的同一车次/航班/线路，直接复述或概括已有内容，无需重新搜索或编造。

            # 答复规范
            请直接以完整、连贯的自然语言进行回复，无需输出后台字段名、程序化标签或固定模板。
            - 车次/航班类：给出车次/航班号、起讫点、发到时刻、历时，以及可得的最相关结果（余票参考、准点指数、联程信息等）。
            - 出行建议类：结合问题给出通俗、可落地、有时效提醒的建议（如提前进站、换乘预留时间），切勿臆造专业结论。
            - 与天气/美食联动：当用户提到“下雨了还能去吗”“路上吃什么”等，可结合出行能力提醒并自然衔接，但不要擅自查询其它模块的实时数据。
            - 日常寒暄：回复亲切简洁，提供积极的情绪价值，并自然引导用户说清出发地、目的地与出行时间。

            # 语言与排版风格
            - **对话口吻**：面向用户直接对话，使用“您”或“你”，亲切自然但不过度轻浮。
            - **排版清晰**：多个并列要点可用 `- ` 无序列表，车次/航班多时用清晰结构呈现。
            - **视觉点缀**：适度使用 emoji（如 🚄、✈️、🗺️）作为情绪点缀，正文中不滥用。
            - **禁忌表述**：严禁出现“无其他问题需要额外回答”、“已按当前时间完成播报”、“详见左侧面板”等元话语、程序化口吻或依赖特定 UI 界面元素的表述。
            - **温暖收尾**：在回答末尾，可自然地附一句轻松的关怀语（如“路上注意安全，祝您出行顺利 🚄”）。

            # 异常处理
            - **工具不可用**：若查询工具返回 None 或不可用，礼貌告知当前系统暂无法获取实时出行数据，并建议用户提供更明确的出发地、目的地与时间后再问。
            - **信息不足（no_data）**：当用户未给出具体出发地与目的地（尤其是把起终点说成"北京市区"这类宽泛区域）时，不要自行假设或编造路线，输出 status="no_data"，在 result 中礼貌说明缺少哪些信息，并引导用户补充具体地点。

            # 输出协议（result / key_data / status）
            你最终必须以下列三个字段输出，缺一不可：
            - `result`：面向用户的完整自然语言回复（车次/航班信息、出行建议或寒暄，即最终给用户看的正文）；当 status="no_data" 时，请用**一句简短、直接的话**告诉用户需要补充什么（如"请告诉我具体的出发地和目的地"），不要冗长铺垫。
            - `key_data`：将本次回答涉及的关键数据提炼为一组英文 snake_case 字段的可读对象（如 train_no/flight_no、from_station、to_station、departure_time、arrival_time、duration、remain_tickets、ticket_office 等）；无相关数据时给空对象 {{}}。
            - `status`：按数据可得性填枚举值 —— 检索到真实数据填 "success"；仅获得部分数据填 "partial"；未获得有效数据但已礼貌向用户说明填 "no_data"；执行过程出错填 "failed"。
            """,
        response_format=ModelOutput,
    )
    try:
        response = await agent.ainvoke(
            {"messages": state["messages"]},
            # 关闭跨图 checkpointer 继承，避免嵌套 agent 误用主图 saver
            {"configurable": {CONFIG_KEY_CHECKPOINTER: None}},
        )
    except Exception:
        # MCP 连接可能失效：清空缓存，下次重新握手
        _tools_cache.invalidate()
        raise
    # 结构化输出缺失时兜底为空 dict
    sr = response.get("structured_response") or {}
    if not isinstance(sr, dict):
        sr = {}
    status = str(sr.get("status", "failed"))
    if status not in {"success", "partial", "no_data", "failed"}:
        status = "success" if sr.get("result") else "failed"
    return {
        "result": str(sr.get("result", "")),
        "key_data": sr.get("key_data", {}) if isinstance(sr.get("key_data"), dict) else {},
        "status": status,
        "evidence": collect_tool_evidence(response.get("messages")),
        "missing_slots": [],
        "retryable": status == "failed",
        "error_code": "" if status != "failed" else "subagent_failed",
    }


def input(state: GraphState):
    return {"messages": HumanMessage(content=state["question"])}


@lru_cache(maxsize=1)
def get_travel_graph():
    graph = StateGraph(GraphState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("travel", travel)
    graph.add_node("input", input)
    graph.add_edge(START, "input")
    graph.add_edge("input", "travel")
    graph.add_edge("travel", END)

    # 子图不配置 checkpointer：状态不持久化，不依赖主图 saver
    return graph.compile()