import os
from dotenv import load_dotenv
import redis

load_dotenv()

# 获取连接
def get_redis_conn():
    # 如果 .env 中配置了 REDIS_PASSWORD 才传，否则不传（避免 Redis 未设密码时报 AUTH 错误）
    redis_password = os.getenv("REDIS_PASSWORD")
    kwargs = {
        "host": os.getenv("REDIS_HOST"),
        "port": int(os.getenv("REDIS_PORT")),
        "db": int(os.getenv("REDIS_DB")),
        # protocol=2 强制使用旧版 RESP2 协议，兼容低版本 Redis（<6.0）
        "protocol": 2,
        # decode_responses=True 自动将 bytes 解码为 str
        "decode_responses": True,
    }
    if redis_password:
        kwargs["password"] = redis_password
    return redis.Redis(**kwargs)


# 关闭连接
def close_redis_conn(conn):
    conn.close()
