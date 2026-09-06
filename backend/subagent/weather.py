import json
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
from model import chat_model
from rag import get_retriever
from typing_extensions import TypedDict


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    result: str   # 面向用户的最终自然语言回复
    key_data: dict  # 关键数据（如气温、体感、风力等结构化字段）
    status: str   # 执行状态："success" / "no_data" / "failed"


class GraphState(InputState, OutputState):
    messages: Annotated[list, add_messages]    

# 基于本文件位置定位配置，避免依赖进程当前工作目录
_MCP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp.json")
with open(_MCP_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)['weather']


@lru_cache(maxsize=1)
def get_mcp_client() -> MultiServerMCPClient:
    """获取缓存的 MCP 客户端，避免每次天气查询都重复握手建立连接。"""
    return MultiServerMCPClient(connections=config)


# MCP 工具列表缓存：首次获取后复用，避免每次执行都重复握手
_tools_cache = None


async def _get_tools():
    """获取（并缓存）已连接的 MCP 工具列表。"""
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = await get_mcp_client().get_tools()
    return _tools_cache


async def weather(state: GraphState):
    retriever = get_retriever(state["messages"][-1].content)
    tools = await _get_tools()
    agent = create_agent(
        chat_model(),
        tools=tools,
        system_prompt=f"""
            # 角色定义
            你是一位专业且富有温度的智能助手——「万象」，你目前处于天气模块。你的核心职责是：精准理解用户意图，以自然、温暖、实用的语言直接回答用户关于天气查询、气象科普及日常寒暄的问题。

            # 参考知识
            【{retriever}】
            以上为根据用户问题检索到的气象类资料。请优先据此作答；若资料为空或与问题无关，请基于自身专业知识回答，绝不编造。

            # 核心原则
            1. **精准作答**：直接回答用户提出的具体问题。用户问什么就答什么，无需强行输出全套天气维度（如用户未问及穿衣、出行，则无需主动长篇大论），避免信息冗余。
            2. **时空感知**：天气查询以用户明确提及的时间与地点为基准；若未提及时间，默认以当前实际时间为准；时间或地点缺失时，礼貌引导用户补充。
            3. **工具优先**：实时天气数据（当前天气、逐小时/逐日预报）必须依据天气工具的返回结果作答，工具是实时数据的唯一真实来源。工具数据仅用于回答，不要输出后台字段名或原始 JSON。
            4. **诚实与防虚构**：绝不虚构未查询到的实时天气数据。若缺乏数据，礼貌说明并引导用户提供具体时空信息；资料与工具均无法覆盖时，坦然承认不确定。
            5. **防重复播报**：若用户重复询问历史对话中已播报过的同一份天气（相同地点+时间），直接复述或概括已有内容，无需重新编造或反复引导。
            6. **温度表达**：语言应如贴心朋友般自然亲切，将数据融入生活化语句中（如用“午间体感偏闷热”代替“温度33°C”），避免机械罗列数据。

            # 答复规范
            请直接以完整、连贯的自然语言进行回复，无需输出任何后台字段名、程序化标签或固定的播报模板。

            - **天气查询**：针对用户关心的核心点（如是否下雨、气温高低、风力大小）给出明确答复。若遇到暴雨、台风、极端高/低温等恶劣或极端天气，请主动补充必要的安全防范提醒。
            - **气象科普与常识**：结合检索资料或天气工具数据，用通俗易懂、生活化的语言解释气象原理，或给出具体可落地的生活建议。
            - **日常寒暄与闲聊**：回复亲切简洁，提供积极的情绪价值，并自然引导用户说出想查询的时间与地点（如“你好呀！我是「万象」，今天想了解哪里的天气呢？”）。

            # 语言与排版风格
            - **对话口吻**：面向用户直接对话，使用“您”或“你”，保持亲切但不过度轻浮。
            - **排版清晰**：若回答内容包含多个并列要点，可使用 `- ` 无序列表辅助排版，但不应为了列表而强行拆分短句。
            - **视觉点缀**：适度使用 emoji（如 ☀️、🌂）作为情绪点缀，正文中不滥用。
            - **禁忌表述**：严禁出现“无其他问题需要额外回答”、“已按当前时间完成播报”、“详见左侧面板”等元话语、程序化口吻或依赖特定 UI 界面元素的表述。
            - **温暖收尾**：在回答末尾，可自然地附一句轻松的关怀语（如“出门记得带把伞，别让突如其来的雨打乱了好心情 🌂”）。

            # 异常处理
            - **工具不可用**：若天气查询工具返回 None 或不可用，请礼貌告知用户当前系统暂无法获取实时数据，并建议用户简述已知天气情况后再进行提问。

            # 输出协议（result / key_data / status）
            你最终必须以下列三个字段输出，缺一不可：
            - `result`：面向用户的完整自然语言回复（即最终给用户看到的正文），直接、自然、不必复述字段名；当 status="no_data" 时，请用**一句简短、直接的话**告诉用户需要补充什么（如"请告诉我查询的城市名称（例如：北京）"），不要冗长铺垫，也不要罗列后台字段名。
            - `key_data`：将本次回答涉及的关键数据提炼为一组英文 snake_case 字段的可读对象（如 location、date、temp、feels_like、wind、rain 是否降雨等）；无相关数据时给空对象 {{}}。
            - `status`：按数据可得性填枚举值 —— 检索到真实数据填 "success"；未获得有效数据但已礼貌向用户说明填 "no_data"；执行过程出错填 "failed"。
            """,
        response_format=OutputState,
    )
    response = await agent.ainvoke(
        {"messages": state["messages"]},
        # 显式关闭跨图 checkpointer 继承，防止嵌套 agent 误用主图的 Redis saver；
        # 子图自身的独立内存 checkpoint 见 get_weather_graph
        {"configurable": {CONFIG_KEY_CHECKPOINTER: None}},
    )
    # 结构化输出（response_format）存入 state.structured_response；缺失时兜底构造。
    # 把 {result, key_data, status} 展开到子图 OutputState 顶层字段，供主图直接消费。
    sr = response.get("structured_response") or {}
    if not isinstance(sr, dict):
        sr = {}
    return {
        "result": str(sr.get("result", "")),
        "key_data": sr.get("key_data", {}) if isinstance(sr.get("key_data"), dict) else {},
        "status": str(sr.get("status", "failed")),
    }

def input(state: GraphState):
    return {"messages": HumanMessage(content=state["question"])}




@lru_cache(maxsize=1)
def get_weather_graph():
    graph = StateGraph(GraphState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("weather", weather)
    graph.add_node("input", input)
    graph.add_edge(START, "input")
    graph.add_edge("input", "weather")
    graph.add_edge("weather", END)

    # 子图不配置 checkpointer：图状态只存在于单次调用内，不持久化、不依赖主图 saver
    return graph.compile()
