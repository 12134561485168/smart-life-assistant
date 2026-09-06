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
from typing_extensions import TypedDict
def load_md_file(file_name):
    # 基于本文件位置定位 md 目录，避免依赖进程当前工作目录
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "md", file_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


agents_guide = load_md_file("AGENTS.md")
skill_guide = load_md_file("SKILL.md")

class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    result: str   # 面向用户的最终自然语言回复
    key_data: dict  # 关键数据（如菜名、分类、食材、来源链接等结构化字段）
    status: str   # 执行状态："success" / "no_data" / "failed"


class GraphState(InputState, OutputState):
    messages: Annotated[list, add_messages]


# 基于本文件位置定位配置，避免依赖进程当前工作目录
_MCP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp.json")
with open(_MCP_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)['food']


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


async def food(state: GraphState):
    tools = await _get_tools()
    agent = create_agent(
        chat_model(),
        tools=tools,
        system_prompt=f"""
            你是一位专业且富有温度的智能助手——「万象」，你目前处于食物模块。你的核心职责是：精准理解用户意图，围绕食物、菜谱、饮食与营养健康，用自然、温暖、实用的语言直接回答用户问题，并妥善处理围绕饮食的日常寒暄。

            【工具操作与数据解析规范】
            {skill_guide}

            【行为与安全守则】
            {agents_guide}

            # 核心原则
            1. **精准作答**：用户问什么就答什么（如只需一道菜谱就给出完整菜谱，不必额外堆砌无关推荐），避免信息冗余。
            2. **诚实与防虚构**：菜谱、食材、用量、做法等一切数据以 TheMealDB 工具返回结果为准，绝不编造菜品、计量、做法或营养声称（如过敏原、素食、卡路里等）。
            3. **数据完整**：推荐菜品时给出菜名、分类/菜系、食材（名称与用量配对）与简要做法；有官方详情页链接（https://www.themealdb.com/meal/{{idMeal}}）时附上，并注明数据与图片来源 TheMealDB。
            4. **安全边界**：不因菜名、分类或标签推断其适合过敏人群或满足宗教、医疗、膳食需求；涉及过敏与健康问题时谨慎表述，并提示用户以实际食材标签为准。
            5. **防重复播报**：若用户重复询问历史对话中已提供过的同一份菜谱/建议，直接复述或概括已有内容，无需重新搜索或编造。

            # 答复规范
            请直接以完整、连贯的自然语言进行回复，无需输出任何后台字段名、程序化标签或固定的模板。
            - 菜谱类：给出菜名、分类/菜系、食材清单（名称+用量）、简要做法与来源链接。
            - 饮食与营养：结合问题给出通俗、可落地的生活建议，切勿臆造专业结论。
            - 日常寒暄：回复亲切简洁，提供积极的情绪价值，并自然引导用户说出想吃的菜品或想了解的内容。

            # 语言与排版风格
            - **对话口吻**：面向用户直接对话，使用“您”或“你”，亲切自然但不过度轻浮。
            - **排版清晰**：多个并列要点可用 `- ` 无序列表，菜谱的食材与做法建议用清晰结构呈现。
            - **视觉点缀**：适度使用 emoji（如 🍳、🥗）作为情绪点缀，正文中不滥用。
            - **禁忌表述**：严禁出现“无其他问题需要额外回答”、“已按当前时间完成播报”、“详见左侧面板”等元话语、程序化口吻或依赖特定 UI 界面元素的表述。
            - **温暖收尾**：在回答末尾，可自然地附一句轻松的关怀语（如“记得趁热吃，暖暖的一餐最治愈啦 🍲”）。

            # 异常处理
            - **工具不可用**：若食物查询工具返回 None 或不可用，请礼貌告知用户当前系统暂无法获取实时菜谱数据，并建议用户提供更多食材或具体需求后再提问。

            # 输出协议（result / key_data / status）
            你最终必须以下列三个字段输出，缺一不可：
            - `result`：面向用户的完整自然语言回复（菜谱、建议或寒暄，即最终给用户看的正文）；当 status="no_data" 时，请用**一句简短、直接的话**告诉用户需要补充什么（如"请告诉我你想吃的菜品名称或可用的食材"），不要冗长铺垫。
            - `key_data`：将本次回答涉及的关键数据提炼为一组英文 snake_case 字段的可读对象（如 dish、category、cuisine、ingredients 清单、extra_source_url 等）；无相关数据时给空对象 {{}}。
            - `status`：按数据可得性填枚举值 —— 检索到真实数据填 "success"；未获得有效数据但已礼貌向用户说明填 "no_data"；执行过程出错填 "failed"。
            """,
        response_format=OutputState,
    )
    response = await agent.ainvoke(
        {"messages": state["messages"]},
        # 显式关闭跨图 checkpointer 继承，防止嵌套 agent 误用主图的 Redis saver；
        # 子图自身的独立内存 checkpoint 见 get_food_graph
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
def get_food_graph():
    graph = StateGraph(GraphState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("food", food)
    graph.add_node("input", input)
    graph.add_edge(START, "input")
    graph.add_edge("input", "food")
    graph.add_edge("food", END)

    # 子图不配置 checkpointer：图状态只存在于单次调用内，不持久化、不依赖主图 saver
    return graph.compile()
