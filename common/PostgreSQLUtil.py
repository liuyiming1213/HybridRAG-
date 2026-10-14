# 使用 pg8000.dbapi 驱动（纯 Python 实现，无 Windows 编码问题，API 兼容 psycopg2）
import pg8000.dbapi


# 获取Postgres连接（对标你原来mysql_conn）
def pg_conn():
    return pg8000.dbapi.connect(
        user="postgres",
        password="13118086790liuyi",
        host="localhost",
        port=5432,
        database="postgres"
    )


# 关闭连接函数（逻辑几乎不变，不用大改）
def pg_close(cur, conn):
    cur.close()
    conn.close()