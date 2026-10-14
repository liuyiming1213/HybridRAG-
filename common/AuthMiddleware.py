"""
认证中间件 — FastAPI 依赖注入方式保护接口

工作原理：
  FastAPI 的 Depends (依赖注入) 是一种 AOP (面向切面编程) 机制。
  把"鉴权"这个横切关注点抽成独立函数，哪个接口需要鉴权，
  就在路由函数参数里声明 Depends(get_current_user)，FastAPI
  会在调用路由函数之前先执行鉴权函数。

使用方式：
  from common.AuthMiddleware import get_current_user

  @router.get("/some-protected-api")
  def protected_endpoint(current_user: dict = Depends(get_current_user)):
      # current_user 就是从 token 中解析出的用户信息
      # 如果 token 无效，请求在进入这里之前就被拦截了
      return {"user": current_user}

前端如何配合：
  每次请求在 HTTP Header 中带上：
    Authorization: Bearer <这里的token>

  具体代码（axios）：
    axios.get('/api/xxx', {
        headers: { 'Authorization': 'Bearer ' + token }
    })
"""

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from common import JWTUtil

# HTTPBearer 是 FastAPI 内置的安全方案，自动从请求头提取 Authorization: Bearer xxx
# auto_error=False 表示没有 token 时先不报错，我们自己控制错误信息
security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> dict:
    """
    从请求头提取并校验 JWT Token，返回当前用户信息。

    这是 FastAPI 的 Depends 函数，用法：
      @router.get("/api/xxx")
      def xxx(user: dict = Depends(get_current_user)):
          ...

    校验流程：
      1. 检查请求头是否有 Authorization: Bearer <token>
      2. 提取 token 字符串
      3. 调用 JWTUtil.verify_access_token() 校验签名+过期+类型
      4. 校验通过 → 返回 payload（包含用户ID和用户名）
      5. 任何一步失败 → 抛出 HTTPException，返回 401 给前端

    返回示例：{"sub": "1", "name": "cc"}
    """
    # Step 1：检查是否有 token
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证令牌，请在请求头中添加 Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials  # 提取 token 字符串（去掉 "Bearer " 前缀）

    # Step 2：校验 token
    try:
        payload = JWTUtil.verify_access_token(token)
        return payload  # 校验通过，返回用户信息
    except jwt.ExpiredSignatureError:
        # Token 过期了 → 前端应该用 refresh_token 去换新的
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证令牌已过期，请使用刷新令牌获取新的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        # Token 无效（被篡改、签名不匹配、类型不对等）
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证令牌无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
