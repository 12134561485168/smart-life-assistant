import os
import sys
from typing import Literal

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Windows 控制台默认 GBK，直接 print 含特殊字符（如分数符号）的中文/Unicode 会崩溃，
# 这里强制 stdout 使用 UTF-8，保证任意控制台编码下都能正常运行
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()  # 加载 .env 文件中的环境变量

# 创建 MCP 实例，对应「MCP 服务器」角色
mcp = FastMCP(
    "meal", host=os.getenv("food_host", "127.0.0.1"), port=int(os.getenv("food_port", "8001"))
)

API_BASE_URL = os.getenv("API_BASE_URL", "https://www.themealdb.com/api/json/v1")
# 建议通过环境变量注入真实的 API Key，此处默认使用测试 Key '1'
API_KEY = os.getenv("THEMEALDB_API_KEY", "1")

async def _make_request(endpoint: str, params: dict) -> dict:
    """
    内部辅助函数：封装 HTTP GET 请求并处理异常。
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/{API_KEY}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            return {"error": "请求失败", "details": str(e)}


@mcp.tool()
async def food_search_meals_by_name(name: str) -> dict:
    """
    根据完整或部分的餐点名称进行搜索。
    :param name: 餐点名称（支持模糊匹配）。
    """
    return await _make_request("search.php", {"s": name})


@mcp.tool()
async def food_search_meals_by_first_letter(letter: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]) -> dict:
    """
    根据餐点名称的首字母进行搜索。
    :param letter: 单个英文字母（A-Z）。
    """
    return await _make_request("search.php", {"f": letter})


@mcp.tool()
async def food_lookup_meal_by_id(meal_id: int) -> dict:
    """
    根据数字 ID 查找餐点的完整详细信息（包含食谱、食材、视频链接等）。
    :param meal_id: 餐点的唯一数字标识符。
    """
    return await _make_request("lookup.php", {"i": meal_id})

@mcp.tool()
async def food_filter_meals(
    ingredient: str | None = None,
    category: str | None = None,
    area: str | None = None
) -> dict:
    """
    根据食材、分类或地区/菜系筛选餐点摘要列表。请且仅请提供以下三个参数中的一个。
    :param ingredient: 食材名称（例如: "chicken_breast"）。
    :param category: 分类名称（例如: "Dessert"）。
    :param area: 地区或菜系名称（例如: "Italian"）。
    """
    params = {}
    if ingredient:
        params["i"] = ingredient
    if category:
        params["c"] = category
    if area:
        params["a"] = area
        
    if len(params) != 1:
        return {"error": "参数校验失败：必须且只能提供 ingredient, category, area 中的一个参数。"}
        
    return await _make_request("filter.php", params)


@mcp.tool()
async def food_list_all_categories() -> dict:
    """获取系统中所有可用的餐点分类名称列表。"""
    return await _make_request("list.php", {"c": "list"})


@mcp.tool()
async def food_list_all_areas() -> dict:
    """获取系统中所有可用的地区/菜系名称列表。"""
    return await _make_request("list.php", {"a": "list"})


@mcp.tool()
async def food_list_all_ingredients() -> dict:
    """获取系统中所有可用的食材详细记录列表。"""
    return await _make_request("list.php", {"i": "list"})


@mcp.tool()
async def food_get_detailed_categories() -> dict:
    """获取包含描述和缩略图链接的详细餐点分类信息。"""
    return await _make_request("categories.php", {})


@mcp.tool()
async def food_get_random_meal() -> dict:
    """随机获取一个餐点的完整详细信息。"""
    return await _make_request("random.php", {})


@mcp.tool()
async def food_get_latest_meals() -> dict:
    """获取数据库中最新添加的餐点列表（结果数量取决于 API 访问级别）。"""
    return await _make_request("latest.php", {})


# 使用示例
if __name__ == "__main__":
    # print(get_location_by_amap('北理工'))
    # lon,lat = get_location_by_amap('北理工')
    # get_weather(lon,lat,'2026-08-26','2026-08-27')
    mcp.run(transport="streamable-http")
