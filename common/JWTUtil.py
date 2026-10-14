"""
JWT 工具模块 — Token 生成、校验、解析

技术原理：
  JWT (JSON Web Token) 由三部分组成，用 . 分隔：
    Header.Payload.Signature

  Header  ：记录签名算法（HS256）和 token 类型（JWT）
  Payload ：存放业务数据（用户ID、用户名、过期时间等）
  Signature：对 Header+Payload 的签名，防止被篡改
             签名密钥只有服务端知道，任何人改 Payload 都会导致签名不匹配

工作流程：
  登录成功 → 服务端生成 JWT → 返回给前端
  前端请求 → 请求头带 Authorization: Bearer <token>：是HTTP协议中传递身份令牌的标准格式
  服务端收到 → 解析 token 校验签名+过期时间 → 通过则放行

为什么选 JWT 而不是 Session？
  - JWT 是无状态的，服务端不用存会话信息，水平扩展方便
  - 适合前后端分离架构，前端存 token，每次请求自己带上
"""

import jwt  # PyJWT 库：pip install PyJWT
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ---- 密钥配置 ----
# SECRET_KEY：JWT 签名密钥，必须保密，泄露后任何人都能伪造 token
# 从 .env 文件读取，如果没配置则用默认值（生产环境务必修改！）
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "worldcup-jwt-secret-key-2026")

# 签名算法：HS256 是最常用的对称加密算法，服务端用同一个密钥加解密
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Token 过期时间
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))   # 访问token 1小时
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_DAYS", "7"))        # 刷新token 7天


def create_access_token(user_id: int, username: str) -> str:
    """
    生成访问 Token（短时效，用于日常 API 鉴权）

    参数：
      user_id  - 用户ID，存入 token 的 payload 中
      username - 用户名，也存入 payload，方便后续日志/展示

    返回：
      签好名的 JWT 字符串，如 "eyJhbGciOi...xxxx"

    Payload 包含：
      sub   : 用户ID（标准字段 subject）
      name  : 用户名
      type  : token 类型标记为 "access"（区分刷新token）
      iat   : 签发时间（issued at）
      exp   : 过期时间（expiration）
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),                                # 用户ID（JWT标准字段）
        "name": username,                                   # 用户名
        "type": "access",                                   # 标记为访问token
        "iat": now,                                         # 签发时间
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),  # 过期时间
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int, username: str) -> str:
    """
    生成刷新 Token（长时效，用于在访问 token 过期后无感续期）

    为什么需要刷新 token？
      - 访问 token 过期时间短（1小时），即使泄露影响范围也有限
      - 刷新 token 过期时间长（7天），前端用它去换新的访问 token
      - 用户不用频繁重新登录，安全性和体验兼顾
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "name": username,
        "type": "refresh",                                                  # 标记为刷新token
        "iat": now,
        "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    解析 JWT Token，校验签名和过期时间

    参数：token - JWT 字符串
    返回：payload 字典，如 {"sub":"1", "name":"cc", "type":"access", ...}
    异常：
      jwt.ExpiredSignatureError → token 已过期
      jwt.InvalidTokenError      → token 无效（被篡改、签名不对等）
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def verify_access_token(token: str) -> dict:
    """
    校验访问 Token 是否合法（签名正确 + 未过期 + type=access）

    流程：
      1. 解码 token（自动验证签名和过期）
      2. 检查 type 是否为 "access"（防刷新token冒充访问token）
      3. 返回 payload 供下游使用

    返回示例：{"sub":"1", "name":"cc"}
    异常：token 无效或类型不对会抛出对应异常
    """
    payload = decode_token(token)

    # 必须是 access token，不允许用 refresh token 访问业务接口
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Token 类型错误：需要 access token，但收到的是 refresh token")

    return payload



#  调试
if __name__ == "__main__":
    # 1. 生成 token
    access = create_access_token(user_id=1, username="cc")
    refresh = create_refresh_token(user_id=1, username="cc")
    print(f"访问Token: {access[:50]}...")
    print(f"刷新Token: {refresh[:50]}...")

    # 2. 校验 token
    payload = verify_access_token(access)
    print(f"校验通过！用户: {payload['name']}(ID:{payload['sub']})")

    # 3. 测试错误 token（刷 refresh token 伪装 access token）
    try:
        verify_access_token(refresh)
    except jwt.InvalidTokenError as e:
        print(f"预期中的错误: {e}")
