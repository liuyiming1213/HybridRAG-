# 导入数据访问层和服务工具
from chat.dao import HistoryDao
from common import ResponseUtil


# 根据用户ID查询历史记录列表（服务层）
# 功能说明：查询指定用户的所有对话根记录，并转换为前端需要的格式
# 参数说明：user_id - 用户ID
# 返回说明：返回包含 code, msg, data 的字典，data 中是历史记录列表
def find_history_by_user_id(user_id):
    # 1. 调用数据访问层查询数据库
    result = HistoryDao.find_history_by_user_id(user_id)

    # 2. 创建空列表存储处理后的数据
    history_list = []

    # 3. 遍历查询结果，转换为前端需要的格式
    for item in result:
        history_list.append({
            # title 用于显示在历史记录列表中，使用问题作为标题
            'title': item['question'],
            # historyId 用于前端标识这条历史记录，对应数据库中的 history_id 字段
            'historyId': item['history_id'],
            # question 和 answer 用于展示详细信息
            'question': item['question'],
            'answer': item['answer'],
            # createTime 格式化为 'YYYY-MM-DD HH:MM:SS' 格式
            'createTime': item['create_time'].strftime('%Y-%m-%d %H:%M:%S')
        })

    # 4. 返回统一的响应格式
    return ResponseUtil.response_json(200, "success", history_list)


# 获取完整的对话记录（服务层）
# 功能说明：根据对话ID获取该对话的所有消息，转换为前端消息格式
# 参数说明：historyId - 对话的根ID
# 返回说明：返回包含 code, msg, data 的字典，data 中是消息列表（每条消息包含 role 和 content）
def find_history_by_id(historyId):
    # 1. 调用数据访问层查询该对话的所有消息
    result = HistoryDao.find_history_by_id(historyId)

    # 2. 创建空列表存储消息
    data_list = []

    # 3. 遍历查询结果，将每条问答转换为两条消息（用户消息和助手消息）
    for item in result:
        # 添加用户消息
        data_list.append({
            "role": "user",           # 角色：用户
            "content": item['question']  # 内容：问题
        })
        # 添加助手消息
        data_list.append({
            "role": "assistant",      # 角色：助手
            "content": item['answer']   # 内容：答案
        })

    # 4. 返回消息列表
    return ResponseUtil.response_json(200, "success", data_list)


# 保存继续对话的结果（服务层）
# 功能说明：在现有对话中新增一条问答记录，或者创建新的对话
# 参数说明：history - ChatHistory 对象，包含 question, answer, parentId, userId
# 返回说明：返回包含 code, msg, data 的字典，data 中是新插入记录的ID
def open_history_save_data(history):
    # 1. 调用数据访问层保存数据到数据库
    result = HistoryDao.open_history_save_data(history)

    # 2. 判断是否保存成功
    if result != 0:
        # 保存成功后，判断是否需要触发上下文压缩
        # 确定对话的根ID：新对话根ID=新插入的ID；继续对话根ID=parentId
        root_id = result if history.parentId == 0 else history.parentId
        # 触发压缩（只有总条数是5的倍数且>5时才真正执行压缩）
        maybe_compress(root_id)

        # 保存成功，返回状态码200和新插入的ID
        return ResponseUtil.response_json(200, "success", result)
    else:
        # 保存失败，返回状态码500
        return ResponseUtil.response_json(500, "error", result)


# 删除历史记录（服务层）
# 功能说明：根据历史记录ID删除该对话的所有消息
# 参数说明：historyId - 要删除的对话ID
# 返回说明：返回包含 code, msg, data 的字典
def delete_history_by_id(historyId):
    # 1. 调用数据访问层删除数据
    result = HistoryDao.delete_history_by_id(historyId)

    # 2. 判断是否删除成功（rowcount > 0 表示有记录被删除）
    if result > 0:
        # 删除成功，联动删除该对话的压缩摘要，避免残留
        HistoryDao.delete_summaries_by_root(historyId)
        return ResponseUtil.response_json(200, "success", "删除成功")
    else:
        # 删除失败，返回状态码500和失败消息
        return ResponseUtil.response_json(500, "error", "删除失败")


# 获取对话记忆上下文（服务层）
def get_chat_memory(historyId, limit=10):
    # 1. 调用数据访问层获取最近N轮对话记录
    memory_list = HistoryDao.get_chat_memory(historyId, limit)

    # 2. 如果没有历史记录，返回空字符串
    if not memory_list:
        return ""

    # 3. 将历史记录格式化为对话格式的文本
    memory_parts = []
    for item in memory_list:
        memory_parts.append(f"用户: {item['question']}")
        memory_parts.append(f"助手: {item['answer']}")

    # 4. 用换行符拼接所有对话
    return "\n".join(memory_parts)


# ================================================================
#  上下文压缩功能（控制 LLM 调用成本）
#
#  设计思路（压缩游标模型）：
#    - 最近 5 条记录永不压缩，保持原始文本
#    - 每当对话总条数达到 5 的倍数（10, 15, 20...）时，
#      把"还没被压缩的最早 5 条"压缩成一段摘要，存进 history_summary 表
#    - 注入 memory 时：所有摘要 + 最近未压缩的记录
#
#  为什么这样设计能降低成本：
#    - 压缩是每 5 条才调一次 LLM，调用频率降为原来的 1/5
#    - 长对话时，早期的几十条记录被压缩成一段摘要，
#      注入的 Token 数大幅减少，节省 LLM 推理成本
# ================================================================

# 判断是否需要压缩，需要就执行压缩（保存对话后调用）
# 功能说明：检查该对话总条数，满足条件就把最早的5条未压缩记录压缩成摘要
# 参数说明：root_history_id - 对话根ID
# 压缩规则：总条数是 5 的倍数且 > 5 时触发
def maybe_compress(root_history_id):
    # 如果是新对话（根ID为0或None），直接跳过
    if not root_history_id or root_history_id == 0:
        return

    # 1. 查该对话总记录数
    total = HistoryDao.count_records(root_history_id)

    # 2. 只有总数 > 5 且是 5 的倍数时才压缩
    #    例如：10 条时压缩 1-5；15 条时压缩 6-10；20 条时压缩 11-15
    if total <= 5 or total % 5 != 0:
        return

    # 3. 查压缩游标（已压缩到哪个 history_id）
    max_compressed = HistoryDao.get_max_compressed_id(root_history_id)

    # 4. 取未压缩的最早 5 条记录（游标之后、最早的5条）
    oldest_5 = HistoryDao.get_oldest_uncompressed(root_history_id, max_compressed, limit=5)

    # 5. 不足5条就不压缩（理论上不会到这里，因为 total 是5的倍数）
    if len(oldest_5) < 5:
        return

    # 6. 调 LLM 把这5条压缩成摘要
    #    压缩只是成本优化，绝不能拖垮保存对话的主流程：
    #    LLM 调用失败（配额、网络、超时）时只打印日志跳过，让保存正常返回
    try:
        summary = _compress_records(oldest_5)
    except Exception as e:
        print(f"[压缩] 对话{root_history_id} 压缩失败，跳过本次：{e}")
        return

    # 7. 保存摘要到 history_summary 表
    #    压缩范围 = 这5条的起始 history_id 到结束 history_id
    HistoryDao.save_summary(
        root_history_id,
        summary,
        oldest_5[0]['history_id'],
        oldest_5[-1]['history_id']
    )
    print(f"[压缩] 对话{root_history_id}：已压缩 {oldest_5[0]['history_id']}-{oldest_5[-1]['history_id']} 共5条")


# 调 LLM 把若干条问答记录压缩成一段摘要
# 功能说明：把多轮问答浓缩成简短摘要，保留关键信息（人名、偏好、结论等）
# 参数说明：records - 问答记录列表
# 返回说明：压缩后的摘要文本
# 成本说明：用非流式 invoke 调用，每5条只调一次，相比每次对话都调LLM省80%
def _compress_records(records):
    from ai.models import LoadALYModel

    # 1. 把记录拼成对话文本
    dialogue_parts = []
    for item in records:
        dialogue_parts.append(f"用户: {item['question']}")
        dialogue_parts.append(f"助手: {item['answer']}")
    dialogue_text = "\n".join(dialogue_parts)

    # 2. 构造压缩 Prompt
    #    要求 LLM 提取关键信息（用户身份、偏好、重要结论），丢弃寒暄和冗余
    compress_prompt = f"""请将以下对话记录压缩成一段简洁的摘要，保留关键信息（如用户姓名、偏好、重要结论、具体问题等），去除寒暄和冗余内容。摘要控制在100字以内。

对话记录：
{dialogue_text}

摘要："""

    # 3. 调用 LLM（非流式）
    llm = LoadALYModel.load_model()
    response = llm.invoke(compress_prompt)
    return response.content


# 获取带摘要的对话记忆（用于注入 Prompt 的 memory）
# 功能说明：把所有摘要 + 最近未压缩的记录 拼成完整记忆文本
# 参数说明：root_history_id - 对话根ID
# 返回说明：完整的记忆文本（摘要在前，最近记录在后）
def get_memory_with_summary(root_history_id):
    # 新对话直接返回空
    if not root_history_id or root_history_id == 0:
        return ""

    # 1. 查压缩游标
    max_compressed = HistoryDao.get_max_compressed_id(root_history_id)

    # 2. 查所有摘要（按时间正序）
    summaries = HistoryDao.get_all_summaries(root_history_id)

    # 3. 查游标之后的记录（最近未压缩的，即最近5条左右）
    uncompressed = HistoryDao.get_records_after_cursor(root_history_id, max_compressed)

    # 4. 拼接：摘要在前 + 最近记录在后
    parts = []
    # 摘要部分
    for s in summaries:
        parts.append(f"【历史摘要】{s['summary']}")
    # 最近未压缩的原始记录
    for r in uncompressed:
        parts.append(f"用户: {r['question']}")
        parts.append(f"助手: {r['answer']}")

    return "\n".join(parts)


# 测试代码：查询用户ID为1的历史记录
if __name__ == '__main__':
    find_history_by_user_id(1)