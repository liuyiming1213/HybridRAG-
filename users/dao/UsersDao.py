# 操作数据库的文件
from common import PostgreSQLUtil


# 根据用户名查询用户信息
def check_email(username):
    # 获取连接对象
    conn = PostgreSQLUtil.pg_conn()
    # 获取游标对象
    cur = conn.cursor()
    # SQL
    sql = "SELECT * FROM users WHERE email=%s;"
    # 执行
    cur.execute(sql, [username])
    # 获取结果和列名
    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    # 关闭连接
    PostgreSQLUtil.pg_close(cur, conn)
    # 转换为字典列表
    return [dict(zip(columns, row)) for row in result]





