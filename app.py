"""
Python 3.14 workarounds - must run before chainlit:
1. anyio doesn't detect the asyncio event loop on 3.14, breaking Starlette's FileResponse.
2. asyncio.timeout() requires current_task(), which fails in engineio/uvicorn.
"""
import asyncio

import anyio.to_thread as _anyio_to_thread

# Patch 1: anyio.to_thread.run_sync -> use asyncio's executor
_original_run_sync = _anyio_to_thread.run_sync
async def _run_sync_via_asyncio(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
_anyio_to_thread.run_sync = _run_sync_via_asyncio

# Patch 2: asyncio.wait_for fallback when timeout() raises "Timeout should be used inside a task"
_original_wait_for = asyncio.wait_for

async def _wait_for_fallback(aw, timeout=None):
    if timeout is None:
        return await aw
    task = asyncio.ensure_future(aw)
    try:
        done, pending = await asyncio.wait([task], timeout=timeout)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise asyncio.TimeoutError()
    finally:
        if not task.done():
            task.cancel()

async def _patched_wait_for(aw, timeout=None, **kwargs):
    try:
        return await _original_wait_for(aw, timeout=timeout, **kwargs)
    except RuntimeError as e:
        if "Timeout should be used inside a task" in str(e):
            return await _wait_for_fallback(aw, timeout)
        raise

asyncio.wait_for = _patched_wait_for

import chainlit as cl
import ollama

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )

    msg = cl.Message(content="")

    start_message = "Hello, I'm your 100% local alternative to ChatGPT running on DeepSeek-R1. How can I help you today?"

    for token in start_message:
        await msg.stream_token(token)

    await msg.send()

@cl.step(type="tool")
async def tool(input_message):

    interaction = cl.user_session.get("interaction")

    interaction.append({"role": "user",
                            "content": input_message})
    
    response = ollama.chat(model="deepseek-r1:8b",
                           messages=interaction) 
    
    interaction.append({"role": "assistant",
                        "content": response.message.content})
    
    return response


@cl.on_message 
async def main(message: cl.Message):

    tool_res = await tool(message.content)

    msg = cl.Message(content="")
    for token in tool_res.message.content:
        await msg.stream_token(token)
        
    await msg.send()