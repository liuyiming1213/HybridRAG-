# 导入历史记录服务和实体类
from chat.service import HistoryService
from chat.dao import HistoryDao
from fastapi import APIRouter, Depends
from chat.entity.ChatHistory import ChatHistory
from common import ResponseUtil
from common.AuthMiddleware import get_current_user

# 创建历史记录路由器
history_router = APIRouter()



#  查询用户历史记录列表（受 JWT 保护）
#  认证方式：Authorization: Bearer <access_token>
#  说明：userId 不再信任前端传参，而是从 token 里解码出来，
#        防止用户伪造 userId 去查别人的历史记录
@history_router.get("/list")
def find_history_by_user_id(current_user: dict = Depends(get_current_user)):
    # 从 token 的 payload 里取出用户ID（sub 字段存的就是用户ID）
    user_id = int(current_user["sub"])
    try:
        return HistoryService.find_history_by_user_id(user_id)
    except Exception as e:
        return ResponseUtil.response_json(500, str(e), None)


#  获取完整的历史对话记录（受 JWT 保护 + 归属校验）
#  说明：查对话详情前，先校验这条记录是否属于当前登录用户
@history_router.get("/findHistoryById")
def find_history_by_id(id: int, current_user: dict = Depends(get_current_user)):
    # 鉴权：查这条记录的归属用户
    owner_id = HistoryDao.find_history_owner(id)
    token_user_id = int(current_user["sub"])
    # 如果记录不存在，或记录不属于当前用户，拒绝访问
    if owner_id is None:
        return ResponseUtil.response_json(404, "对话记录不存在", None)
    if owner_id != token_user_id:
        return ResponseUtil.response_json(403, "无权访问该对话记录", None)
    try:
        return HistoryService.find_history_by_id(id)
    except Exception as e:
        return ResponseUtil.response_json(500, str(e), None)


#  保存对话记录（受 JWT 保护）
#  说明：userId 不再信任前端传参，而是从 token 里解码出来，
#        确保保存的记录一定归属当前登录用户
@history_router.post("/openHistorySaveData")
def open_history_save_data(history: ChatHistory, current_user: dict = Depends(get_current_user)):
    # 用 token 里的用户ID 覆盖前端传来的 userId，防止伪造
    history.userId = int(current_user["sub"])
    try:
        return HistoryService.open_history_save_data(history)
    except Exception as e:
        return ResponseUtil.response_json(500, str(e), None)


#  删除历史记录（受 JWT 保护 + 归属校验）
@history_router.delete("/delete")
def delete_history(historyId: int, current_user: dict = Depends(get_current_user)):
    # 鉴权：查这条记录的归属用户
    owner_id = HistoryDao.find_history_owner(historyId)
    token_user_id = int(current_user["sub"])
    # 如果记录不存在，或记录不属于当前用户，拒绝删除
    if owner_id is None:
        return ResponseUtil.response_json(404, "对话记录不存在", None)
    if owner_id != token_user_id:
        return ResponseUtil.response_json(403, "无权删除该对话记录", None)
    try:
        return HistoryService.delete_history_by_id(historyId)
    except Exception as e:
        return ResponseUtil.response_json(500, str(e), None)
