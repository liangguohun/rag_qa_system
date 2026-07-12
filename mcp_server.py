# mcp_server.py - 使用标准 MCP API
import asyncio
import datetime
import math
import json
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, GetPromptResult, Prompt, PromptArgument

# 创建服务器实例
server = Server("local-tools-server")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """列出所有可用工具"""
    return [
        Tool(
            name="get_current_time",
            description="获取当前时间，支持不同格式",
            inputSchema={
                "type": "object",
                "properties": {
                    "format_type": {
                        "type": "string",
                        "enum": ["full", "date", "time"],
                        "description": "时间格式类型",
                        "default": "full"
                    }
                }
            }
        ),
        Tool(
            name="calculate",
            description="安全计算数学表达式，例如：2+3*4",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="get_word_length",
            description="统计输入文本的字符长度",
            inputSchema={
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "要统计的文本"
                    }
                },
                "required": ["word"]
            }
        ),
        Tool(
            name="weather_check",
            description="查询指定城市天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                        "default": "北京"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    try:
        if name == "get_current_time":
            format_type = arguments.get("format_type", "full")
            now = datetime.datetime.now()
            
            if format_type == "full":
                formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            elif format_type == "date":
                formatted = now.strftime("%Y-%m-%d")
            elif format_type == "time":
                formatted = now.strftime("%H:%M:%S")
            else:
                formatted = now.strftime("%Y-%m-%d %H:%M:%S")
            
            return [TextContent(type="text", text=f"当前时间: {formatted}")]
            
        elif name == "calculate":
            expression = arguments.get("expression", "")
            try:
                allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
                allowed_names.update({"abs": abs, "round": round})
                code = compile(expression, "<string>", "eval")
                for name in code.co_names:
                    if name not in allowed_names:
                        raise NameError(f"禁止使用{name}")
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return [TextContent(type="text", text=f"计算结果: {expression} = {result}")]
            except Exception as e:
                return [TextContent(type="text", text=f"计算异常: {str(e)}")]
                
        elif name == "get_word_length":
            word = arguments.get("word", "")
            return [TextContent(type="text", text=f"'{word}' 字符长度: {len(word)}")]
            
        elif name == "weather_check":
            city = arguments.get("city", "北京")
            weather_map = {
                "北京": "晴天，25°C，湿度45%",
                "上海": "多云，28°C，湿度60%",
                "广州": "小雨，30°C，湿度75%",
                "深圳": "阴天，29°C，湿度70%"
            }
            weather = weather_map.get(city, "晴天，22°C，湿度50%")
            return [TextContent(type="text", text=f"{city}: {weather}")]
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        return [TextContent(type="text", text=f"执行错误: {str(e)}")]

async def main():
    """启动服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="local-tools-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())