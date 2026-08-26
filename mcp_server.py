# mcp_server.py - 使用 FastMCP API（标准 MCP Server）
import datetime
import math
import sys
from mcp.server.fastmcp import FastMCP

# MCP 使用 stdin/stdout 传输 JSON-RPC，print() 会污染协议，日志必须走 stderr
def _log(msg: str):
    sys.stderr.write(f"[MCP_SERVER] {msg}\n")
    sys.stderr.flush()

# 创建服务器实例
server = FastMCP("local-tools-server")


@server.tool()
async def get_current_time(format_type: str = "full") -> str:
    """获取当前时间，支持不同格式（full / date / time）"""
    _log(f"call_tool 被调用! 工具名=get_current_time, 参数=format_type:{format_type}")
    now = datetime.datetime.now()
    if format_type == "date":
        formatted = now.strftime("%Y-%m-%d")
    elif format_type == "time":
        formatted = now.strftime("%H:%M:%S")
    else:
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间: {formatted}"


@server.tool()
async def calculate(expression: str) -> str:
    """安全计算数学表达式，例如：2+3*4"""
    _log(f"call_tool 被调用! 工具名=calculate, 参数={expression}")
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round})
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"禁止使用{name}")
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算异常: {str(e)}"


@server.tool()
async def get_word_length(word: str) -> str:
    """统计输入文本的字符长度"""
    _log(f"call_tool 被调用! 工具名=get_word_length, 参数={word}")
    return f"'{word}' 字符长度: {len(word)}"


@server.tool()
async def weather_check(city: str = "北京") -> str:
    """查询指定城市天气"""
    _log(f"call_tool 被调用! 工具名=weather_check, 参数={city}")
    weather_map = {
        "北京": "晴天，25°C，湿度45%",
        "上海": "多云，28°C，湿度60%",
        "广州": "小雨，30°C，湿度75%",
        "深圳": "阴天，29°C，湿度70%"
    }
    weather = weather_map.get(city, "晴天，22°C，湿度50%")
    return f"{city}: {weather}"


if __name__ == "__main__":
    server.run()
