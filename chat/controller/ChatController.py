import json
from starlette.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from chat.service import ChatService
from fastapi import APIRouter, Request
from common import JWTUtil, ResponseUtil

chat_router = APIRouter()
templates = Jinja2Templates(directory="templates")


# 流式聊天（支持对话记忆 + JWT 认证）
# 请求示例：GET /chat/chatStream?question=你好&historyId=4&token=xxx
# 参数说明：
#   question  - 用户问题
#   historyId - 当前对话的根ID，用于加载对话记忆
#   token     - JWT access_token

# 为什么 token 走 URL 参数而不是 Authorization 头？
#   SSE 使用浏览器原生 EventSource，它不支持自定义请求头，
#   无法在 Header 里加 Authorization: Bearer xxx。
#   所以只能把 token 拼在 URL 的查询参数里传递。
#   这是 SSE 方案的已知权衡（token 会暴露在 URL 和访问日志中），
#   更安全的生产方案是用 fetch + ReadableStream 替代 EventSource。
@chat_router.get("/chatStream") #Depend失败时不兼容SSE
def chat_stream(question: str, historyId: int = 0, token: str = ""):
    # 校验 JWT，失败直接返回错误（SSE 格式）
    try:
        payload = JWTUtil.verify_access_token(token)
    except Exception:
        # token 无效或过期，返回 401 错误提示（以 SSE 格式）
        def error_generator():
            yield f"data: {json.dumps({'content': '[ERROR] 认证失败，请重新登录'})}\n\n"
            yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
        return StreamingResponse(content=error_generator(), media_type="text/event-stream")

    def generator():
        # 将 historyId 传给 ChatService，让它加载对话记忆
        for chunk in ChatService.chat_stream(question, historyId):
            yield f"data: {json.dumps({'contet': chunk})}\n\n"
        yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"

    return StreamingResponse(
        content=generator(),
        media_type="text/event-stream"
    )


