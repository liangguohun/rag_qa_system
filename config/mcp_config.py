# config/mcp_config.py
from pathlib import Path
import json
import sys
import os

ROOT = Path(__file__).parent.parent

# MCP服务器配置
MCP_SERVERS = {
    "local-tools": {
        "command": sys.executable,
        "args": ["mcp_server.py"],
        "transport": "stdio",
        # "enabled": True # MultiServerMCPClient 的配置参数不支持 enabled 字段
    },
    # 可以添加更多MCP服务器配置
    # "remote-tools": {
    #     "url": "http://localhost:8000/mcp",
    #     "transport": "sse",
    #     "enabled": False
    # }
}


# 工具启用/禁用配置（独立管理）
ENABLED_SERVERS = {
    "local-tools": True,
    # "remote-tools": False
}

# 工具启用/禁用配置
ENABLED_TOOLS = {
    "get_current_time": True,
    "calculate": True,
    "get_word_length": True,
    "weather_check": True
}

# 从环境变量或配置文件加载
def load_mcp_config(config_file: str = None):
    """
    从配置文件加载MCP配置
    
    Returns:
        tuple: (servers_config, enabled_servers)
    """
    servers_config = MCP_SERVERS.copy()
    enabled_servers = ENABLED_SERVERS.copy()
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            
            # 分离服务器配置和启用状态
            for name, config in user_config.items():
                if isinstance(config, dict):
                    # 如果有enabled字段，保存到enabled_servers
                    if "enabled" in config:
                        enabled_servers[name] = config.pop("enabled")
                    
                    # 更新服务器配置（移除enabled字段）
                    servers_config[name] = config
    
    return servers_config, enabled_servers

def get_enabled_servers(servers_config, enabled_servers):
    """
    获取启用的服务器配置
    
    Args:
        servers_config: 服务器配置字典
        enabled_servers: 启用状态字典
    
    Returns:
        dict: 启用的服务器配置
    """
    return {
        name: config 
        for name, config in servers_config.items() 
        if enabled_servers.get(name, True)  # 默认启用
    }


# 添加环境变量支持
def get_mcp_config_from_env():
    """从环境变量获取MCP配置"""
    config = {}
    
    # 从环境变量读取MCP服务器配置
    mcp_servers_env = os.getenv("MCP_SERVERS", "")
    if mcp_servers_env:
        try:
            config = json.loads(mcp_servers_env)
        except:
            pass
    
    # 读取工具启用配置
    enabled_tools = os.getenv("ENABLED_TOOLS", "")
    if enabled_tools:
        try:
            enabled = json.loads(enabled_tools)
            ENABLED_TOOLS.update(enabled)
        except:
            pass
    
    return config

# 初始化时合并环境变量配置
MCP_SERVERS.update(get_mcp_config_from_env())