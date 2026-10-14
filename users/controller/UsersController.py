from fastapi import APIRouter, Depends
from users.service import UsersService
from users.entity.CheckCodeEntity import CheckCodeEntity
from common import JWTUtil
from common.AuthMiddleware import get_current_user

# 子路由对象
users_router = APIRouter()



#  发送验证码 — GET /users/sendEmail
@users_router.get("/sendEmail",
    summary="发送邮箱验证码",
    description="向指定邮箱发送4位数字验证码，60秒有效")
def send_email(email: str):
    return UsersService.send_email(email)



#  验证码登录 — POST /users/checkCode
#  验证通过后同样返回 JWT Token
@users_router.post("/checkCode",
    summary="验证验证码登录",
    description="验证码正确后返回 JWT access_token + refresh_token")
def check_code(checkCodeEntity: CheckCodeEntity):
    return UsersService.check_code(checkCodeEntity)



#  Token 刷新 — POST /users/refresh
#  前端在 access_token 过期后，用 refresh_token 换取新的 access_token
#  这样用户不用重新登录，体验无感续期
@users_router.post("/refresh",
    summary="刷新访问令牌",
    description="用 refresh_token 换取新的 access_token（旧refresh_token不变）")
def refresh_token(refresh_token: str):
    """
    刷新 Token 流程：
      1. 前端检测到 401（token过期）
      2. 自动用 refresh_token 调这个接口
      3. 服务端校验 refresh_token 的签名和过期时间
      4. 校验通过 → 签发新的 access_token
      5. 前端拿到新 access_token，重试之前失败的请求
    """
    try:
        payload = JWTUtil.decode_token(refresh_token)
        if payload.get("type") != "refresh":
            return {"code": 500, "msg": "Token类型错误，需要refresh_token", "data": None}
        user_id = int(payload["sub"])
        username = payload["name"]
        new_access = JWTUtil.create_access_token(user_id, username)
        return {"code": 200, "msg": "令牌刷新成功", "data": {"access_token": new_access}}
    except Exception as e:
        return {"code": 500, "msg": f"刷新令牌无效或已过期: {str(e)}", "data": None}


#  受保护接口示例 — GET /users/me
#  需要 Authorization: Bearer <access_token> 才能访问
#  用于测试 JWT 认证中间件是否正常工作
@users_router.get("/me",
    summary="获取当前用户信息（需认证）",
    description="需要请求头 Authorization: Bearer <access_token>，返回当前登录用户信息")
def get_my_info(current_user: dict = Depends(get_current_user)):
    """
    这个接口展示了如何在任意路由上添加 JWT 认证保护。
    只需要在函数参数里加 Depends(get_current_user) 即可。

    未登录访问 → 返回 401
    已登录访问 → 返回用户信息
    """
    return {
        "code": 200,
        "msg": "认证成功",
        "data": {
            "user_id": current_user["sub"],
            "username": current_user["name"],
        },
    }


def api_router():
    return None