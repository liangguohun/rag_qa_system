"""
mariadb_mcp_service.py
本地 MariaDB MCP 服务（HTTP）的启停管理，供 main.py / multiAgent/mariadb_langgraph_agent.py 复用。

由 config/settings.py 的 MARIADB_MCP_TRANSPORT 开关决定：
    http  —— 伴随启动：拉起本地 vendor/mariadb-mcp 的 HTTP Service，等待端口就绪后返回进程
    stdio —— 延后启动：不启动外部服务，由 Agent 调用时内嵌拉起 stdio 子进程，此处返回 None
"""
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

# 确保项目根目录在 sys.path 中，以导入 config 包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import (
    MARIADB_MCP_TRANSPORT,
    MARIADB_DB_CONFIG,
    MARIADB_MCP_READ_ONLY,
    MARIADB_MCP_HTTP_HOST,
    MARIADB_MCP_HTTP_PORT,
    MARIADB_MCP_HTTP_PATH,
    MARIADB_MCP_HTTP_URL,
    MARIADB_MCP_DIR,
)


def get_transport() -> str:
    """当前传输模式开关（settings.py 手动指定）：http / stdio"""
    t = MARIADB_MCP_TRANSPORT.lower()
    if t not in ("http", "stdio"):
        raise SystemExit(f"不支持的传输模式: {t}（settings.py 中可选 http / stdio）")
    return t


def get_mcp_url() -> str:
    """http 模式下的 MCP 端点地址"""
    return (MARIADB_MCP_HTTP_URL
            or f"http://{MARIADB_MCP_HTTP_HOST}:{MARIADB_MCP_HTTP_PORT}{MARIADB_MCP_HTTP_PATH}")


async def _port_open(host: str, port: int) -> bool:
    """探测端口是否已有服务在监听"""
    try:
        _reader, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, ConnectionError):
        return False


async def _verify_mcp_ready(host: str, port: int, path: str, timeout: float = 3.0) -> bool:
    """发 MCP initialize 握手，确认服务真实可用。

    端口能建立 TCP 连接 ≠ MCP 协议可用：残留的半死进程（能 accept 但立即
    关闭/不响应）会让 _port_open 误判为"已有服务"，复用后 Agent 真实请求
    才会暴露 ConnectError。这里用一次真实握手验证。
    """
    url = f"http://{host}:{port}{path}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "rag_qa_system", "version": "1.0"},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
        return resp.status_code < 500 and resp.headers.get("mcp-session-id") is not None
    except Exception:
        return False


async def _kill_port_owner(port: int) -> None:
    """杀掉占用指定端口的进程（Windows 下连带进程树），用于替换半死残留进程"""
    if sys.platform != "win32":
        return
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return
    pids = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line.upper():
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    for pid in pids:
        if pid.isdigit():
            _ = subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)


async def _wait_port(host: str, port: int, timeout: float = 60.0) -> None:
    """轮询等待端口就绪"""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if await _port_open(host, port):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"等待 MCP HTTP Server {host}:{port} 就绪超时")


async def _wait_mcp_ready(
    host: str, port: int, path: str, timeout: float = 60.0
) -> bool:
    """轮询等待 MCP initialize 握手成功。

    服务进程启动后 TCP 端口可能先开，但协议层（数据库连接池等）还需
    数秒才就绪；只握手一次易误判失败而误杀进程。这里在超时内持续轮询。
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if await _verify_mcp_ready(host, port, path):
            return True
        await asyncio.sleep(1.0)
    return False


async def _terminate_process(proc: asyncio.subprocess.Process | None) -> None:
    """结束子进程（Windows 下连带终止整个进程树）"""
    if proc is None or proc.returncode is not None:
        return
    if sys.platform == "win32":
        _ = subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
        )
    else:
        proc.terminate()
        try:
            _ = await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()


async def _pump(stream: asyncio.StreamReader) -> None:
    """把子进程 stdout 转发到 stderr，避免管道缓冲阻塞子进程"""
    while True:
        line = await stream.readline()
        if not line:
            break
        _ = sys.stderr.write(f"[mcp-http-server] {line.decode(errors='replace')}")
        _ = sys.stderr.flush()


def _launch_cmd(mcp_dir: Path) -> list[str]:
    """用仓库自带 venv 的 python 启动本地 MCP Server；无 venv 时回退 uv run"""
    if sys.platform == "win32":
        venv_python = mcp_dir / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = mcp_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python), "src/server.py"]
    return ["uv", "--directory", str(mcp_dir), "run", "src/server.py"]


async def ensure_http_server() -> asyncio.subprocess.Process | None:
    """http 模式：保证本地 MariaDB MCP HTTP Service 在运行。
    端口已占用且握手验证通过 → 直接复用；否则杀掉残留进程自动拉起并等待就绪。"""
    if await _port_open(MARIADB_MCP_HTTP_HOST, MARIADB_MCP_HTTP_PORT):
        if await _verify_mcp_ready(MARIADB_MCP_HTTP_HOST, MARIADB_MCP_HTTP_PORT, MARIADB_MCP_HTTP_PATH):
            print(f"[mariadb-mcp] 检测到可用的 MCP HTTP Service: {get_mcp_url()}，直接复用")
            return None
        print("[mariadb-mcp] 端口被占用但服务不可用（疑似残留/半死进程），替换为新实例 ...")
        await _kill_port_owner(MARIADB_MCP_HTTP_PORT)
        await asyncio.sleep(2)

    mcp_dir = Path(MARIADB_MCP_DIR)
    if not (mcp_dir / "pyproject.toml").exists():
        raise RuntimeError(
            f"http 模式需要本地 MariaDB MCP 仓库：{mcp_dir}\n"
            + "  请先部署：git clone https://github.com/MariaDB/mcp " + str(mcp_dir) + "\n"
            + "  并在该目录执行 uv sync 初始化依赖"
        )
    cmd = _launch_cmd(mcp_dir) + [
        "--transport", "http",
        "--host", MARIADB_MCP_HTTP_HOST,
        "--port", str(MARIADB_MCP_HTTP_PORT),
        "--path", MARIADB_MCP_HTTP_PATH,
    ]
    print(f"[mariadb-mcp] 自动拉起本地 MariaDB MCP HTTP Service: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(mcp_dir),
        env={**os.environ, **MARIADB_DB_CONFIG, "MCP_READ_ONLY": MARIADB_MCP_READ_ONLY},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout:
        _ = asyncio.create_task(_pump(proc.stdout))
    try:
        await _wait_port(MARIADB_MCP_HTTP_HOST, MARIADB_MCP_HTTP_PORT)
        if not await _wait_mcp_ready(MARIADB_MCP_HTTP_HOST, MARIADB_MCP_HTTP_PORT, MARIADB_MCP_HTTP_PATH):
            raise RuntimeError(f"MCP HTTP Service 端口已开但握手失败: {get_mcp_url()}")
        print(f"[mariadb-mcp] MCP HTTP Service 已就绪: {get_mcp_url()}")
        return proc
    except Exception:
        # 注意：uv run 可能已退出，proc.pid 指向已消失的父进程，taskkill 杀不到
        # 真正监听端口的是孤儿 python 子进程，必须按端口 netstat 定位后清理
        await _terminate_process(proc)
        await _kill_port_owner(MARIADB_MCP_HTTP_PORT)
        raise


async def start_mariadb_mcp_service() -> asyncio.subprocess.Process | None:
    """按 settings.py 开关管理本地 MariaDB MCP 服务（伴随 / 延后启动）：
       http  → 伴随启动本地 HTTP Service，返回其进程（main 退出时需 stop 清理）
       stdio → 延后启动，返回 None（由 Agent 调用时内嵌拉起，无需外部服务）"""
    if get_transport() == "http":
        print("[mariadb-mcp] http 模式：伴随启动本地 MariaDB MCP HTTP Service ...")
        return await ensure_http_server()
    print("[mariadb-mcp] stdio 模式：延后启动，由 Agent 调用时内嵌拉起 MCP 子进程，无需外部服务")
    return None


async def stop_mariadb_mcp_service(proc: asyncio.subprocess.Process | None) -> None:
    """停止伴随启动的本地 MariaDB MCP HTTP Service（未启动则无操作）"""
    if proc is None:
        return
    await _terminate_process(proc)
    print("[mariadb-mcp] 本地 MariaDB MCP HTTP Service 已停止")
