# 操作数据库的文件
from common import PostgreSQLUtil


# 根据用户ID查询历史记录列表（只查 parent_id=0 的根记录）
def find_history_by_user_id(user_id):
    # 1. 获取PostgreSQL数据库连接对象
    conn = PostgreSQLUtil.pg_conn()
    # 2. 创建游标对象，用于执行SQL语句
    cur = conn.cursor()
    # 3. 编写SQL查询语句
    # 说明：LEFT JOIN 关联 users 表，通过 history_fk_users 字段连接
    # WHERE 条件：u.id=%s 匹配用户ID，AND h.parent_id=0 只查询根记录（parent_id=0 表示对话的开始）
    sql = '''SELECT h.* FROM "history" h LEFT JOIN "users" u ON h.history_fk_users=u.id WHERE u.id=%s AND h.parent_id=0 ORDER BY h.create_time DESC;'''
    # 4. 执行SQL语句，传入参数 user_id
    cur.execute(sql, [user_id])
    # 5. 获取查询结果，返回的是元组列表
    result = cur.fetchall()
    # 6. 获取列名列表，用于后续转换为字典格式
    columns = [desc[0] for desc in cur.description]
    # 7. 将元组列表转换为字典列表，方便前端使用
    # zip(columns, row) 将列名和值配对，dict() 转换为字典
    result = [dict(zip(columns, row)) for row in result]
    # 8. 提交事务（虽然查询不需要提交，但规范上保留）
    conn.commit()
    # 9. 关闭游标和连接，释放数据库资源
    PostgreSQLUtil.pg_close(cur, conn)
    # 10. 返回查询结果
    return result

# 根据历史记录ID获取完整的对话记录（包括所有子记录）
def find_history_by_id(historyId):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    # 查询 history_id=historyId（根记录）或 parent_id=historyId（子记录）的所有记录
    sql = '''SELECT "question", "answer", "history_id", "create_time" FROM "history" WHERE "history_id"=%s OR "parent_id"=%s ORDER BY "create_time" ASC;'''
    cur.execute(sql, [historyId, historyId])
    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in result]
    conn.commit()
    PostgreSQLUtil.pg_close(cur, conn)
    return result

# 保存继续对话的结果
def open_history_save_data(history):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    try:
        sql = '''INSERT INTO "history" ("question", "answer", "create_time", "parent_id", "history_fk_users") VALUES (%s,%s,now(),%s,%s) RETURNING history_id;'''
        cur.execute(sql, [history.question, history.answer, history.parentId, history.userId])
        conn.commit()
        new_history_id = cur.fetchone()[0]
        return new_history_id
    except Exception as e:
        print(e)
        conn.rollback()
        return 0
    finally:
        PostgreSQLUtil.pg_close(cur, conn)

# 查询历史记录的归属用户（用于权限鉴权）
def find_history_owner(historyId):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    sql = '''SELECT "history_fk_users" FROM "history" WHERE "history_id"=%s;'''
    cur.execute(sql, [historyId])
    result = cur.fetchone()
    PostgreSQLUtil.pg_close(cur, conn)
    if result:
        return result[0]
    return None


# 删除历史记录及其所有子记录
def delete_history_by_id(historyId):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    try:
        sql = '''DELETE FROM "history" WHERE "history_id"=%s OR "parent_id"=%s;'''
        cur.execute(sql, [historyId, historyId])
        conn.commit()
        return cur.rowcount
    except Exception as e:
        print(e)
        conn.rollback()
        return 0
    finally:
        PostgreSQLUtil.pg_close(cur, conn)


# 获取对话记忆上下文（用于AI记忆功能）
# 功能说明：根据对话根ID，查询该对话下的最近N条问答记录，供AI作为记忆上下文
def get_chat_memory(historyId, limit=10):
    # 如果 historyId 为0或None，说明是新对话，没有历史上下文
    if not historyId or historyId == 0:
        return []

    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    # 查询该对话根记录及其所有子记录（parent_id=historyId），按时间倒序取最近N条
    sql = '''SELECT "question", "answer" FROM (
                 SELECT "question", "answer", "create_time" FROM "history"
                 WHERE "history_id"=%s OR "parent_id"=%s
                 ORDER BY "create_time" DESC
                 LIMIT %s
             ) AS recent ORDER BY "create_time" ASC;'''
    cur.execute(sql, [historyId, historyId, limit])
    # 获取查询结果
    result = cur.fetchall()#元组只有值，没有字段名
    # 获取列名列表
    columns = [desc[0] for desc in cur.description]
    # 转换为字典列表
    result = [dict(zip(columns, row)) for row in result]
    conn.commit()
    PostgreSQLUtil.pg_close(cur, conn)
    return result


# ================================================================
#  摘要压缩相关（对话记忆压缩功能）
# ================================================================

# 查询该对话已经压缩到哪个 history_id（压缩游标）
# 功能说明：查 history_summary 表里该对话最大的 end_history_id，
#          表示"压缩到这条为止"，游标之前的都已经被压缩成摘要了
# 参数说明：root_history_id - 对话的根ID
# 返回说明：返回已压缩的最大 history_id，没压缩过则返回 0
def get_max_compressed_id(root_history_id):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    sql = '''SELECT COALESCE(MAX(end_history_id), 0) FROM "history_summary" WHERE "root_history_id"=%s;'''
    cur.execute(sql, [root_history_id])
    result = cur.fetchone()
    PostgreSQLUtil.pg_close(cur, conn)
    return result[0] if result else 0


# 查询该对话的总记录数（根记录 + 所有子记录）
# 功能说明：判断是否需要触发压缩的依据
# 参数说明：root_history_id - 对话的根ID
# 返回说明：返回该对话的总条数
def count_records(root_history_id):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    sql = '''SELECT COUNT(*) FROM "history" WHERE "history_id"=%s OR "parent_id"=%s;'''
    cur.execute(sql, [root_history_id, root_history_id])
    result = cur.fetchone()
    PostgreSQLUtil.pg_close(cur, conn)
    return result[0] if result else 0


# 查询未压缩的最早 N 条记录（用于压缩）
# 功能说明：查 history_id > 压缩游标 的记录中，最早的一批（用于拿去压缩）
# 参数说明：root_history_id - 对话根ID；max_compressed - 压缩游标；limit - 取几条
# 返回说明：按时间正序排列的记录列表
def get_oldest_uncompressed(root_history_id, max_compressed, limit=5):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    # history_id 或 parent_id 属于这个对话，且 history_id > 游标（还没压缩过）
    # 按 history_id 正序取最早的 limit 条（create_time 是 date 类型，同一天排序不稳定，故用 history_id）
    sql = '''SELECT "question", "answer", "history_id", "create_time" FROM "history"
             WHERE ("history_id"=%s OR "parent_id"=%s) AND "history_id">%s
             ORDER BY "history_id" ASC LIMIT %s;'''
    cur.execute(sql, [root_history_id, root_history_id, max_compressed, limit])
    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in result]
    PostgreSQLUtil.pg_close(cur, conn)
    return result


# 查询游标之后的所有记录（最近未压缩的，用于注入 memory）
# 功能说明：查 history_id > 压缩游标 的所有记录（即还没被压缩的最近记录）
# 参数说明：root_history_id - 对话根ID；max_compressed - 压缩游标
# 返回说明：按时间正序排列的未压缩记录列表
def get_records_after_cursor(root_history_id, max_compressed):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    sql = '''SELECT "question", "answer" FROM "history"
             WHERE ("history_id"=%s OR "parent_id"=%s) AND "history_id">%s
             ORDER BY "history_id" ASC;'''
    cur.execute(sql, [root_history_id, root_history_id, max_compressed])
    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in result]
    PostgreSQLUtil.pg_close(cur, conn)
    return result


# 保存一条摘要
# 功能说明：把压缩后的摘要存入 history_summary 表
# 参数说明：root_history_id - 对话根ID；summary - 摘要文本
#          start_id - 压缩范围起始history_id；end_id - 压缩范围结束history_id（游标）
def save_summary(root_history_id, summary, start_id, end_id):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    try:
        sql = '''INSERT INTO "history_summary" ("root_history_id", "summary", "start_history_id", "end_history_id", "create_time")
                 VALUES (%s, %s, %s, %s, now()) RETURNING id;'''
        cur.execute(sql, [root_history_id, summary, start_id, end_id])
        conn.commit()
        new_id = cur.fetchone()[0]
        return new_id
    except Exception as e:
        print(e)
        conn.rollback()
        return 0
    finally:
        PostgreSQLUtil.pg_close(cur, conn)


# 查询该对话的所有摘要（按时间正序）
# 功能说明：注入 memory 时，把所有摘要放在最前面
# 参数说明：root_history_id - 对话根ID
# 返回说明：摘要列表，每条含 summary
def get_all_summaries(root_history_id):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    sql = '''SELECT "summary" FROM "history_summary" WHERE "root_history_id"=%s ORDER BY "create_time" ASC;'''
    cur.execute(sql, [root_history_id])
    result = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    result = [dict(zip(columns, row)) for row in result]
    PostgreSQLUtil.pg_close(cur, conn)
    return result


# 删除该对话的所有摘要（删除对话时联动删除）
# 功能说明：删除历史记录时，对应摘要也要删掉，避免残留
# 参数说明：root_history_id - 对话根ID
# 返回说明：删除的行数
def delete_summaries_by_root(root_history_id):
    conn = PostgreSQLUtil.pg_conn()
    cur = conn.cursor()
    try:
        sql = '''DELETE FROM "history_summary" WHERE "root_history_id"=%s;'''
        cur.execute(sql, [root_history_id])
        conn.commit()
        return cur.rowcount
    except Exception as e:
        print(e)
        conn.rollback()
        return 0
    finally:
        PostgreSQLUtil.pg_close(cur, conn)


if __name__ == "__main__":
    # 测试：查询用户ID为1的所有历史记录
    print(find_history_by_user_id(1))