# test_mcp_client.py - 使用绝对路径
import asyncio
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_tools():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    # 构建 MCP 服务器脚本的绝对路径
    server_script = script_dir.parent / "mcp_server.py"
    server_script = server_script.resolve()
    
    print(f"🔍 MCP 服务器路径: {server_script}")
    print(f"🔍 文件是否存在: {server_script.exists()}")
    
    if not server_script.exists():
        print(f"❌ 找不到 MCP 服务器文件!")
        return
    
    try:
        client = MultiServerMCPClient({
            "local-tools": {
                "command": sys.executable,
                "args": [str(server_script)],  # 使用绝对路径
                "transport": "stdio"
            }
        })
        
        print("🔄 正在连接 MCP 服务器...")
        tools = await client.get_tools()
        print(f"✅ 成功加载 {len(tools)} 个工具:")
        
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
        
        # 测试工具
        if tools:
            print("\n🔄 测试工具调用...")
            result = await tools[0].ainvoke({"format_type": "full"})
            print(f"📅 结果: {result}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tools())