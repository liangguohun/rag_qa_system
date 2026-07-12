# mcp_server.py
from mcp.server.fastmcp import FastMCP
import datetime
import math

# 初始化MCP服务
mcp = FastMCP("LocalToolsServer")

# 1. 获取当前时间工具
@mcp.tool()
def get_current_time(format_type: str = "full") -> str:
    """
    获取当前时间
    Args:
        format_type: 可选 full/date/time
    """
    now = datetime.datetime.now()
    if format_type == "date":
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        return now.strftime("%H:%M:%S")
    return now.strftime("%Y-%m-%d %H:%M:%S")

# 2. 数学计算工具
@mcp.tool()
def calculate(expression: str) -> str:
    """安全计算数学表达式，例如：2+3*4"""
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round})
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"禁止使用{name}")
        res = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果:{expression}={res}"
    except Exception as e:
        return f"计算异常:{str(e)}"

# 3. 文本长度工具
@mcp.tool()
def get_word_length(word: str) -> str:
    """统计输入文本的字符长度"""
    return f"'{word}' 字符长度：{len(word)}"

# 4. 天气模拟工具
@mcp.tool()
def weather_check(city: str = "北京") -> str:
    """查询指定城市天气"""
    weather_map = {
        "北京": "晴天，25°C，湿度45%",
        "上海": "多云，28°C，湿度60%",
        "广州": "小雨，30°C，湿度75%",
        "深圳": "阴天，29°C，湿度70%"
    }
    return f"{city}:{weather_map.get(city, '晴天，22°C，湿度50%')}"

if __name__ == "__main__":
    # 以stdio方式启动MCP服务，供LangChain客户端调用
    mcp.run(transport="stdio")