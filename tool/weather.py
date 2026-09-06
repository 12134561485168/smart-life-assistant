import json
import os
import sys
import time

import httpx
import openmeteo_requests
import pandas as pd
import requests_cache

from mcp.server.fastmcp import FastMCP
from retry_requests import retry
from dotenv import load_dotenv

# Windows 控制台默认 GBK，强制 stdout 使用 UTF-8，避免输出中文/Unicode 时崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()  # 加载 .env 文件中的环境变量
# 创建 MCP 实例，对应「MCP 服务器」角色
mcp = FastMCP(
    "weather", host=os.getenv("weather_host", "127.0.0.1"), port=int(os.getenv("weather_port", "8000"))
)


@mcp.tool()
def get_time():
    """获取当前的北京时间。

    适用于用户询问"现在几点""今天日期""当前时间"等需要时间信息的场景。

    Returns:
        str: 格式为 "YYYY-MM-DD HH:MM:SS" 的北京时间字符串，如 "2026-08-27 14:30:00"。
    """
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    return formatted_time


@mcp.tool()
def weather_get_weather(address: str, start_date: str, end_date: str):
    """根据地址（中文地名或详细地址）查询天气，返回当前天气与指定日期区间的逐小时天气预报。

    参数:
        address (str): 地址描述，如 "北京理工大学" 或 "北京市海淀区中关村南大街5号"。
        start_date (str): 查询起始日期，格式必须为 "2026-08-18"。
        end_date (str): 查询结束日期（含当天），格式必须为 "2026-08-19"。

    注意:
        - API 按美国时间解析入参，输出统一为北京时间。
          若要获取 2026-08-19 全天完整的天气，应传入 start_date="2026-08-18"、
          end_date="2026-08-19"。
        - 返回值为 JSON 字符串，顶层含 "当前天气"（当前时刻）与 "逐小时预报"
          （日期时间、温度、湿度、体感温度、降水概率/降水量、风速、风向等）。
    """
    lat, lon = get_location_by_amap(address)
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "snow_depth",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "is_day",
            "apparent_temperature",
        ],
        "start_date": start_date,
        "end_date": end_date,
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    current = response.Current()
    # 当前天气键名中文化
    current_weather = {
        "日期时间": pd.to_datetime(current.Time(), unit="s", utc=True)
        .tz_convert("Asia/Shanghai")
        .tz_localize(None)
        .isoformat(),
        "2米温度": current.Variables(0).Value(),
        "2米相对湿度": current.Variables(1).Value(),
        "是否白天": current.Variables(2).Value(),
        "体感温度": current.Variables(3).Value(),
    }

    hourly = response.Hourly()

    # 提取 Numpy 数组并直接转换为 Python List 以便 JSON 序列化
    # 逐小时预报变量键名中文化
    hourly_vars = {
        "2米温度": hourly.Variables(0).ValuesAsNumpy().tolist(),
        "2米相对湿度": hourly.Variables(1).ValuesAsNumpy().tolist(),
        "体感温度": hourly.Variables(2).ValuesAsNumpy().tolist(),
        "降水概率": hourly.Variables(3).ValuesAsNumpy().tolist(),
        "降水量": hourly.Variables(4).ValuesAsNumpy().tolist(),
        "降雨量": hourly.Variables(5).ValuesAsNumpy().tolist(),
        "阵雨量": hourly.Variables(6).ValuesAsNumpy().tolist(),
        "降雪量": hourly.Variables(7).ValuesAsNumpy().tolist(),
        "积雪深度": hourly.Variables(8).ValuesAsNumpy().tolist(),
        "10米风速": hourly.Variables(9).ValuesAsNumpy().tolist(),
        "10米风向": hourly.Variables(10).ValuesAsNumpy().tolist(),
        "10米阵风": hourly.Variables(11).ValuesAsNumpy().tolist(),
    }

    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True)
        .tz_convert("Asia/Shanghai")
        .tz_localize(None),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
        .tz_convert("Asia/Shanghai")
        .tz_localize(None),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    hourly_vars["日期时间"] = [t.isoformat() for t in times]

    # 将列式字典转换为行式字典列表 (List of Dictionaries)，即标准的 JSON 数组结构
    # zip(*hourly_vars.values()) 会将每一行的数据打包在一起
    hourly_forecast = [
        dict(zip(hourly_vars, row)) for row in zip(*hourly_vars.values())
    ]

    # 3. 组装最终结果（顶层键名中文化）
    result = {"当前天气": current_weather, "逐小时预报": hourly_forecast}

    return json.dumps(result, ensure_ascii=False)


def get_location_by_amap(address):
    """根据地址（中文地名或详细地址）查询经纬度坐标。

    适用于用户仅提到地点名称（如"北京理工大学"）、需要先解析坐标
    再调用 get_weather 查询该地天气的场景。

    参数:
        address (str): 地址描述，如 "北京理工大学" 或 "北京市海淀区中关村南大街5号"。

    Returns:
        tuple: (纬度, 经度) 二元组浮点数，可直接作为 get_weather 的 lat、lon 参数。
    """
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"address": address, "key": os.getenv("amap_key"), "output": "JSON"}

    # 设置合理的超时时间
    response = httpx.get(url, params=params, timeout=10)
    data = response.json()
    location = data["geocodes"][0]["location"]
    longitude, latitude = location.split(",")
    return float(latitude), float(longitude)


# 使用示例
if __name__ == "__main__":
    # print(get_location_by_amap('北理工'))
    # lon,lat = get_location_by_amap('北理工')
    # get_weather(lon,lat,'2026-08-26','2026-08-27')
    mcp.run(transport="streamable-http")
