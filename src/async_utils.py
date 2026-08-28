# -*- coding: utf-8 -*-
"""
asyncio 工具：根治 Windows 下 stdio 子进程 transport 的
"RuntimeError: Event loop is closed" GC 警告。

背景
----
asyncio.run() 返回即关闭事件循环。若 stdio 子进程（MCP stdio server）
的 BaseSubprocessTransport 尚未完成关闭（close() 的 _call_connection_lost
回调仍排在循环队列里），Python GC 回收 transport 时 __del__ 会在已关闭的
循环上调用 loop.call_soon()，抛 RuntimeError，并打印：

    Exception ignored in: <function BaseSubprocessTransport.__del__ ...>
    RuntimeError: Event loop is closed

两个手段
--------
1. install_subprocess_safe_close()：进程级兜底 patch。
   让 close() 在事件循环已关闭时静默返回（幂等、无副作用），
   任何 asyncio.run / uvicorn 关闭循环后的 GC 场景都不再打印该 traceback。

2. run_coro()：替代裸 asyncio.run()。
   记录本次循环创建的所有 subprocess transport，在关闭循环前主动 close()
   并跑完关闭回调，避免残留孤儿 mcp_server 子进程（根治，不依赖时序）。

用法
----
    from src.async_utils import run_coro
    result = run_coro(some_async_function())
"""

import asyncio

_patched = False


def install_subprocess_safe_close():
    """（兜底）让 BaseSubprocessTransport.close() 在事件循环已关闭时静默忽略。"""
    global _patched
    if _patched:
        return
    _patched = True
    try:
        from asyncio import base_subprocess
    except Exception:
        return
    _orig = getattr(base_subprocess.BaseSubprocessTransport, "close", None)
    if _orig is None or getattr(_orig, "_cb_safe_close", False):
        return

    def _safe_close(self):
        try:
            _orig(self)
        except RuntimeError:
            # 事件循环已关闭，无法再调度回调（纯 GC 清理场景），静默忽略
            pass

    _safe_close._cb_safe_close = True
    base_subprocess.BaseSubprocessTransport.close = _safe_close


# 模块被 import 时自动安装兜底（进程级，一次即可覆盖所有入口）
install_subprocess_safe_close()


def run_coro(coro):
    """替代 asyncio.run()：运行协程，并在关闭事件循环前主动关闭残留子进程。

    返回值与 asyncio.run() 一致（coroutine 的返回值）。
    若 coroutine 抛异常，异常照常向上抛出（清理仍会执行）。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 记录该循环创建的所有 subprocess transport，便于在关闭前统一清理
    _transports = []
    _orig_subprocess_exec = loop.subprocess_exec

    async def _tracking_subprocess_exec(protocol_factory, program, *args, **kwargs):
        transport, protocol = await _orig_subprocess_exec(
            protocol_factory, program, *args, **kwargs
        )
        _transports.append(transport)
        return transport, protocol

    loop.subprocess_exec = _tracking_subprocess_exec

    try:
        return loop.run_until_complete(coro)
    finally:
        # 1) 循环仍存活：关闭所有子进程 transport，并跑完其关闭回调
        for _t in _transports:
            try:
                _t.close()
            except Exception:
                pass
        for _ in range(3):
            try:
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                break
        # 2) 关闭默认线程池（asyncio.run 的等价清理）
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        # 3) 最后才关闭循环
        loop.close()
        asyncio.set_event_loop(None)
