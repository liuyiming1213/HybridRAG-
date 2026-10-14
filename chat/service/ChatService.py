import os
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from ai.models import LoadALYModel
from chat.service import HistoryService
from chat.util import ChromaUtil
import numpy as np


#  项目根路径 & 模型路径
path_base = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 重排序模型路径
reranker_model_name = os.path.abspath(os.path.join(path_base, 'models', 'bge-reranker-large_v1'))
# 向量数据库集合的名字
vector_db_name = 'worldcup_kb'



#  中文停用词表 —— 用于 BM25 分词后过滤无实义词
#  包含：虚词、介词、连词、副词、代词、语气词、量词、方位词等
STOP_WORDS = set([
    "的", "了", "之", "乎", "者", "也", "底", "得", "地", "着", "过", "罢", "而已", "与否",
    # 介词
    "在", "于", "对", "对于", "关于", "把", "被", "将", "跟", "同", "与", "向", "往", "从", "自",
    "由", "以", "凭", "用", "为", "给", "距", "离", "比", "按", "按照",
    # 连词
    "和", "及", "以及", "并", "并且", "同时", "而", "然而", "但是", "但", "却", "不过", "虽然",
    "虽说", "尽管", "如果", "假如", "要是", "倘若", "一旦", "只要", "只有", "无论", "不管",
    "还是", "或者", "要么", "因此", "所以", "因而", "于是", "故此", "再者", "另外", "此外",
    # 无实义副词
    "不", "没", "没有", "勿", "非", "皆", "全", "都", "统统", "全部", "凡", "凡是", "每", "各",
    "甚", "很", "非常", "极", "太", "更", "越", "稍微", "略微", "几乎", "差不多", "仅仅", "只",
    "只是", "就", "便", "才", "方才", "刚刚", "已经", "曾经", "正在", "将要", "即将", "马上",
    "立刻", "忽然", "突然", "时常", "常常", "经常", "偶尔", "从不", "始终", "终究", "反正",
    "大概", "大约", "或许", "好像", "似乎", "未必", "一定", "必定",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "大家", "众人", "本人", "自己",
    "别人", "人家", "这", "那", "此", "该", "某", "某个", "哪些", "哪个", "什么", "何", "谁",
    "哪儿", "哪里", "怎样", "如何", "这么", "那样", "如此",
    # 语气感叹词
    "啊", "呀", "吧", "呢", "吗", "么", "噢", "哦", "唉", "哎", "哼", "哈", "哇", "哟", "啦",
    "咯", "呗", "也罢", "也好", "算了",
    # 量词、泛数字
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿",
    "个", "些", "点", "份", "条", "件", "次", "回", "趟", "种", "类", "样", "般", "多少",
    "几", "全部", "一切", "任何", "所有",
    # 方位时间泛词
    "上", "下", "左", "右", "前", "后", "里", "外", "中", "间", "旁", "边", "内", "中间",
    "当前", "如今", "现在", "过去", "将来", "以后", "之前", "今日", "昨日", "明天", "当下", "此刻", "彼时",
    # 判断动词
    "是", "乃", "有", "无"])



#  分词器 —— 用 jieba 分词 + 停用词过滤
#  功能：将中文文本切分为有意义的词语列表，过滤掉停用词和单字
#  用途：为 BM25 关键词检索提供 tokenize 后的词列表
def tokenize(text):
    """
    jieba 分词 + 停用词过滤
    参数：text - 待分词的中文文本
    返回：过滤后的词语列表（长度 > 1 且不在停用词表中）
    """
    return [word for word in jieba.cut(text)
            if word.strip() and word not in STOP_WORDS and len(word.strip()) > 1]



#  BM25 索引缓存（最大的性能瓶颈优化）
#
#  问题：之前每次请求都从 ChromaDB 取全量文档 → jieba 分词 → 构建 BM25Okapi
#       10,185 篇文档约 2~3 秒，占整个 RAG 流程 60%+ 的耗时
#
#  解决：模块级缓存 + 文档数变更检测
#       - 首次请求时构建索引，缓存在全局变量中
#       - 后续请求直接复用缓存（0 秒）
#       - 检测到文档数变化（知识库被重建）时自动重建

# 缓存结构
_bm25_cache = {
    "docs": None,       # Document 对象列表
    "bm25": None,       # BM25Okapi 实例
    "doc_count": 0,     # 缓存时的文档总数（用于检测知识库是否更新）
}


def get_bm25_cached(vector_db):
    """
    获取 BM25 检索器（带缓存）。

    首次调用：从 ChromaDB 读取全量文档 → jieba 分词 → 构建 BM25 → 缓存
    后续调用：直接返回缓存（0 耗时）
    知识库更新：检测到文档数变化 → 自动重建缓存

    参数：vector_db - ChromaDB 连接对象
    返回：(doc_objects, bm25)
    """
    global _bm25_cache

    # 检测知识库是否更新：比较当前文档数和缓存的文档数
    current_count = vector_db._collection.count()
    if current_count == _bm25_cache["doc_count"] and _bm25_cache["bm25"] is not None:
        # 缓存命中，直接返回
        return _bm25_cache["docs"], _bm25_cache["bm25"]

    # 缓存未命中或知识库已更新，重新构建
    print(f"  [BM25] 构建索引中... ({current_count} 篇文档, 约需 2~3 秒)")
    all_data = vector_db.get()
    ids = all_data['ids']
    documents = all_data['documents']
    metadatas = all_data.get('metadatas')

    # jieba 分词构建语料库
    tokenize_corpus = [tokenize(text) for text in documents]

    # 包装 Document 对象（保留 id 用于 RRF 去重；带上 metadata 使其与向量路一致，
    # 否则融合后 source 等来源信息会丢失，影响调试与来源溯源）
    doc_objects = [
        Document(page_content=text, id=ids[idx],
                 metadata=metadatas[idx] if metadatas else {})
        for idx, text in enumerate(documents)
    ]

    # 构建 BM25 索引
    bm25 = BM25Okapi(tokenize_corpus)

    # 写入缓存
    _bm25_cache["docs"] = doc_objects
    _bm25_cache["bm25"] = bm25
    _bm25_cache["doc_count"] = current_count
    print(f"  [BM25] 索引构建完成，已缓存")

    return doc_objects, bm25


#  BM25 关键词检索
#  功能：对用户问题进行分词，计算与各文档的 BM25 相关性分数，返回 Top-K 文档
#  参数：bm25 - BM25Okapi 实例
#        bm25_docs - Document 对象列表
#        question - 用户问题字符串
#        top_k - 返回的文档数量，默认10
#  返回：Top-K 相关 Document 对象列表
def bm25_search(bm25, bm25_docs, question, top_k=15):
    # 1. 对用户问题分词，计算与每个文档的 BM25 分数
    bm25_scores = bm25.get_scores(tokenize(question))

    # 2. 将分数从高到低排序，取 Top-K 的索引
    sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

    # 3. 按分数降序遍历，同时做内容去重（同一场比赛从两队视角各存一条，
    #    前200字相同=内容相同，只取第一条，避免重复文档浪费检索槽位）
    seen_fps = set()
    result = []
    for idx in sorted_indices:
        doc = bm25_docs[idx]
        fp = doc.page_content[:200]  # 前200字做内容指纹
        if fp not in seen_fps:
            seen_fps.add(fp)
            result.append(doc)
            if len(result) >= top_k:
                break

    return result



#  RRF（Reciprocal Rank Fusion）融合算法（加权版）
#  功能：将向量检索和 BM25 检索的结果进行融合排序
#  原理：对每篇文档，累加它在两个检索列表中的倒数排名分数
#        RRF_score = 向量分 + BM25权重 × BM25分
#        其中单项分 = 1 / (k + rank_i)，k=60（平滑参数）
#
#  为什么给 BM25 更高权重？
#  事实性查询（如"1978年冠军是谁"、"阿根廷赛程"）中，BM25 关键词匹配
#  远优于向量语义检索。向量检索适合"意思相近"的泛化查询，但对精确事实、
#  年份、队名等专有名词的区分能力较弱。加权后 BM25 的好结果不会被向量平均化淹没。
#
#  参数：vector_result - 向量检索结果（Document 列表，含 id）
#        bm25_result   - BM25 检索结果（Document 列表，含 id）
#        top_k         - 最终保留的文档数，默认10
#        bm25_weight   - BM25 权重系数，默认2.0（BM25 分数翻倍，
#                        使关键词匹配的文档在融合后更靠前）
def rrf_fusion(vector_result, bm25_result, top_k=15, bm25_weight=1.5):
    """
    RRF 加权融合 + 内容去重。

    内容去重为什么放在这里而不是 reranker 里？
    - 向量库里同一场比赛从两队视角各存一条，UUID 不同但内容相同
    - 如果只在 reranker 再去重，重复文档已经占掉了融合 Top-N 的坑位
    - 在 RRF 阶段就去重，能腾出位置给真正不同的文档（如 1982、2022 的冠军摘要）
    - 去重策略：取前200字符做"内容指纹"，指纹相同的只保留 RRF 分数最高的那份
    """
    # 先用 UUID 去重，再叠加内容去重
    # Step 1：UUID 去重（同一文档在向量和 BM25 都出现时，分数累加）
    uuid_docs = {}   # {uuid: Document}
    uuid_score = {}   # {uuid: RRF分数}

    for rank, doc in enumerate(vector_result):
        uid = doc.id
        # 向量路返回的 Document 带 metadata（如 source），BM25 路由 get_bm25_cached 构造时未带 metadata，
        # 若两路都命中同一 UUID，应保住向量副本的 metadata，故用 setdefault（不覆盖已存在项）
        uuid_docs.setdefault(uid, doc)
        uuid_score[uid] = uuid_score.get(uid, 0) + 1.0 / (60 + rank + 1)

    for rank, doc in enumerate(bm25_result):
        uid = doc.id
        uuid_docs.setdefault(uid, doc)
        uuid_score[uid] = uuid_score.get(uid, 0) + bm25_weight / (60 + rank + 1)

    # Step 2：内容去重（同一条比赛从两队视角各存一条，UUID 不同但内容几乎一样）
    # 取前200字做指纹，指纹相同 = 内容相同，只保留分数最高的那个
    content_docs = {}  # {指纹: Document}
    content_score = {}  # {指纹: 最高RRF分数}

    for uid, score in sorted(uuid_score.items(), key=lambda x: x[1], reverse=True):
        doc = uuid_docs[uid]
        fp = doc.page_content[:200]  # 前200字做内容指纹
        if fp not in content_docs or score > content_score[fp]:
            content_docs[fp] = doc
            content_score[fp] = score

    # Step 3：按分数降序，取 Top-K
    sorted_fps = sorted(content_score, key=content_score.get, reverse=True)[:top_k]

    return [content_docs[fp] for fp in sorted_fps]



#  意图判断 —— 决定是否需要走 RAG 检索流程
#  为什么不能只用关键词？
#  关键词会漏召回："今夏足球盛宴"、"上一届谁拿的冠军"没有"世界杯"关键词
#  但显然在问世界杯。嵌入相似度能理解语义，不依赖字面匹配。
#  为什么不用额外的小模型？
#  BGE 模型已经在向量检索中加载了，直接复用，零额外模型开销。
#  每次意图判断只需 1 次嵌入计算（~5ms CPU），远快于全 RAG 流程。
#  策略：
#    1. 将用户问题向量化
#    2. 与 4 个世界杯锚点文本的嵌入做余弦相似度
#    3. 最高相似度 > 0.5 → RAG；否则 → 跳过


# 锚点文本（用于判断问题是否属于世界杯领域）
INTENT_ANCHORS = [
    "世界杯赛程查询",
    "FIFA世界排名",
    "足球球队分析",
    "世界杯历史冠军",
    "世界杯比分结果",
]

# 缓存锚点向量，首次计算后复用
_anchor_cache = None


def should_use_rag(question: str, embedding_model) -> bool:
    
    global _anchor_cache
    try:
        q_vec = np.array(embedding_model.embed_query(question))
        if _anchor_cache is None:
            _anchor_cache = [np.array(v) for v in embedding_model.embed_documents(INTENT_ANCHORS)]
        max_sim = max(
            float(np.dot(q_vec, a) / (np.linalg.norm(q_vec) * np.linalg.norm(a)))
            for a in _anchor_cache
        )
        return max_sim > 0.5
    except Exception:
        # 兜底：出错时默认走 RAG（宁可多检，不可漏答）
        return True


# 流式聊天（带对话记忆 + 混合检索 + 意图判断）
def chat_stream(question, historyId=0):
    # 加载历史对话记忆（压缩摘要 + 最近未压缩记录，降低长对话注入 Token 成本）
    memory_text = HistoryService.get_memory_with_summary(historyId) if historyId and historyId > 0 else ""

    # ---- 意图判断：是否需要走 RAG 检索? ----
    embedding_model = HuggingFaceEmbeddings(
        model_name=os.path.abspath(os.path.join(path_base, 'models', 'embedding_model')),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    need_rag = should_use_rag(question, embedding_model)
    if not need_rag:
        # 闲聊模式：不需要检索知识库，但依然保留对话记忆
        # AI 用自身知识 + 历史记忆回复，不用强制拉回世界杯
        llm = LoadALYModel.load_model()
        chat_prompt = PromptTemplate(
            template="""你是一个2026世界杯中文问答助手，友好且专业。

            你正在进行闲聊对话。虽然用户的问题与世界杯赛事无关，但你应正常回应用户，
            记住对话中的信息（如用户告诉你的名字、偏好等），后续用户可以追问这些信息。

            【历史对话】— 其中可能包含用户之前告诉你的事情，如果相关请主动引用
            {memory}

            【用户问题】
            {question}

            请友好地回应用户。如果用户问的问题你确实可以帮助（比如天气、常识等），就正常回答。
            只有在完全无法处理时才委婉提醒用户你的专长是世界杯赛事问答。
            """,
            input_variables=["memory", "question"]
        )
        chain = chat_prompt | llm | StrOutputParser()
        for chunk in chain.stream({"memory": memory_text, "question": question}):
            if chunk:
                yield chunk
        return

    # ---- RAG 模式：完整检索流程 ----
    # 准备检索器（向量 + BM25 双路召回）
    vector_db = ChromaUtil.get_kb_conn()
    # 2.2 创建向量检索器（语义相似度检索，Top-15）
    # k=15 而非 10，因为知识库中 96% 是比赛记录，冠军摘要仅 22 篇(0.2%)
    # k 太小会导致小众但重要的数据源（冠军/排名/赛程）被淹没
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    # 2.3 获取 BM25 检索器（带缓存，首次构建后后续请求零耗时）
    bm25_docs, bm25 = get_bm25_cached(vector_db)

    # ---- 混合检索函数（向量 + BM25 + RRF 融合） ----
    def merge_retrieval(question):
        # 第1路：向量语义检索（ChromaDB余弦相似度）
        vector_result = vector_retriever.invoke(question)
        # 向量结果也去重（同BM25，前200字指纹去重）
        seen_v = set()
        vector_unique = []
        for d in vector_result:
            fp = d.page_content[:200]
            if fp not in seen_v:
                seen_v.add(fp)
                vector_unique.append(d)
        retrieved_docs("向量检索结果(去重后)", vector_unique)

        # 第2路：BM25关键词检索（内置内容去重）
        bm25_result = bm25_search(bm25, bm25_docs, question, 15)
        retrieved_docs("BM25检索结果(去重后)", bm25_result)

        # 第3步：RRF加权融合（内置UUID去重+内容去重，最终输出15条完全不同的文档）
        rrf_docs = rrf_fusion(vector_unique, bm25_result, 15)
        retrieved_docs("rrf融合检索结果(去重后)", rrf_docs)
        return rrf_docs

    # 4. 加载大语言模型
    llm = LoadALYModel.load_model()

    # 5. 构建提示词模板（世界杯中文版 — Markdown 输出）
    # Prompt 包含 {memory}（历史对话）和 {context}（知识库文档）
    template = """
        你是一个2026世界杯球队分析问答助手，用中文对球队进行深度分析。

        【输出格式 — 必须使用 Markdown】

        你的回答必须使用以下 Markdown 格式排版，让用户阅读舒适：

        1. 用 `###` 标题分段（如 `### 赛程安排`、`### 历史战绩`、`### 综合分析`）
        2. 用 `- ` 或 `1. ` 列出多条数据，每条一行
        3. 用 `**粗体**` 标注关键信息（队名、日期、比分、排名等）
        4. 用 `> ` 引用块突出重要结论
        5. 对比类问题使用表格 `| 列1 | 列2 |` 格式
        6. 球队名格式：`**阿根廷(Argentina)**`
        7. 日期格式：`2026年6月17日`
        8. 找不到的信息说 `*暂无数据*`

        【示例 Few-shot】

        ---
        示例1：查询赛程

        用户问题：阿根廷的小组赛赛程

        助手回答：
        ### 阿根廷(Argentina) 小组赛赛程

        阿根廷位于 **Group J**，共有3场小组赛：

        | 日期 | 对阵 | 场地 |
        |------|------|------|
        | 2026年6月17日 09:00 | **阿根廷(Argentina)** vs 阿尔及利亚(Algeria) | Kansas City Stadium |
        | 2026年6月23日 01:00 | 奥地利(Austria) vs **阿根廷(Argentina)** | Dallas Stadium |
        | 2026年6月28日 10:00 | **阿根廷(Argentina)** vs 约旦(Jordan) | Dallas Stadium |

        > 阿根廷的小组赛对手均为非传统强队，出线形势较为有利。
        ---
        示例2：查询历史冠军

        用户问题：1978-2022世界杯冠军分别是谁

        助手回答：
        ### 历届世界杯冠军（1978-2022）

        | 年份 | 冠军 | 亚军 | 金靴 |
        |------|------|------|------|
        | 1978 | **阿根廷(Argentina)** | 荷兰(Netherlands) | Mario Kempes(6球) |
        | 1982 | **意大利(Italy)** | 西德 | Paolo Rossi(6球) |
        | 1986 | **阿根廷(Argentina)** | 西德 | Gary Lineker(6球) |
        | 2022 | **阿根廷(Argentina)** | 法国(France) | Kylian Mbappe(8球) |

        > 阿根廷在近5届世界杯中3次夺冠(1978、1986、2022)，是同期成绩最好的球队。巴西同期仅1冠(1994)，*1982年冠军数据暂无*。
        ---

        【历史对话】
        {memory}

        【知识库上下文】
        {context}

        【用户问题】
        {question}

        请按上述 Markdown 格式规范，用中文直接输出最终答案（不要在答案中写"提取信息""组织归纳"等推理步骤）。
    """
    # 注意：input_variables 要和模板中的 {placeholder} 一一对应
    prompt = PromptTemplate(template=template, input_variables=["memory", "context", "question"])

    # 打印检索到的文档内容（调试用）
    # 功能：在控制台输出向量检索/BM25检索/融合后的文档列表，便于调试和观察检索效果
    # 参数：title - 打印标题（如"向量检索结果"），docs - Document对象列表
    # 返回：原样返回 docs，不修改内容，供下游链节点继续使用
    def retrieved_docs(title, docs):
        print(f"\n{'='*20} {title} {'='*20}")
        if not docs:
            print("  (无文档)")
            return docs
        for i, doc in enumerate(docs, 1):
            print(f"\n【文档 {i}】(id: {doc.id})")
            preview = doc.page_content[:500]
            if len(doc.page_content) > 500:
                preview += "..."
            print(f"  内容: {preview}")
            print("-" * 60)
        return docs

    # 重排序模型
    reranker_model = FlagReranker(
        model_name_or_path=reranker_model_name,
        use_fp16=True,
        local_files_only=True,
        devices="cpu"
    )

    # 重排序
    def reranker(inputs):
        """
        自定义重排序函数
        输入: {"context": List[Document], "question": str, "memory": str}
        输出: {"context": str, "question": str, "memory": str} (直接适配 PromptTemplate)
        """
        docs = inputs["context"]
        question = inputs["question"]
        memory = inputs["memory"]  # 历史对话文本，直接透传不做处理

        print(f"\n🔄 开始进行重排序... (候选文档数: {len(docs)})")

        if not docs:
            print("⚠️ 未检索到任何文档，跳过重排序")
            return {"context": "", "question": question, "memory": memory}

        # 1. 构造 query-document 对并计算相关性分数
        pairs = [[question, doc.page_content] for doc in docs]
        scores = reranker_model.compute_score(pairs)

        # 2. 将分数注入 metadata 并按分数降序排序
        for doc, score in zip(docs, scores):
            doc.metadata["relevance_score"] = round(float(score), 4)
        sorted_docs = sorted(docs, key=lambda d: d.metadata["relevance_score"], reverse=True)

        # 3. 截取 Top-N 高相关文档送入 LLM
        top_n = 5
        final_docs = sorted_docs[:top_n]

        # 4. 打印重排序后的结果
        print(f"✅ 重排序完成，保留 Top-{top_n} 文档：")
        for i, doc in enumerate(final_docs, 1):
            score = doc.metadata.get("relevance_score", "N/A")
            print(f"\n【文档 {i}】(相关度: {score})")
            print(f"id:{doc.id}")
            print(f"  内容: {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
            print("-" * 60)

        # 5. 将文档列表格式化为 context 字符串，供 Prompt 使用
        context_str = "\n\n".join(
            f"[来源{i + 1}] {doc.page_content}" for i, doc in enumerate(final_docs)
        )
        # 6. 返回包含 memory 的字典，memory 原样透传
        return {"context": context_str, "question": question, "memory": memory}

    # 构造问答链
    # RunnableParallel 同时运行三个分支，将结果合并为一个字典
    # context 分支：混合检索（向量 + BM25 + RRF融合），由 merge_retrieval 闭包执行
    # question 分支：RunnablePassthrough 直接透传用户问题字符串
    # memory 分支：RunnableLambda 闭包返回预加载的历史对话文本
    qa_chain = (
            RunnableParallel({
                # 混合检索（向量语义 + BM25关键词 + RRF融合），结果传给下游 reranker
                "context": RunnableLambda(merge_retrieval),
                # 用户问题原样透传
                "question": RunnablePassthrough(),
                # 历史对话记忆（闭包注入，lambda忽略输入直接返回预加载值）
                "memory": RunnableLambda(lambda x: memory_text),
            })
            | RunnableLambda(reranker)  # 重排序：从融合结果中精选 Top-3
            | prompt                    # 填充提示词模板
            | llm                        # 大模型推理（流式输出）
            | StrOutputParser()          # 解析输出为纯文本字符串
    )

    # 传入用户问题字符串（链内部通过 RunnablePassthrough 透传，再加上闭包注入的 memory）
    for chunk in qa_chain.stream(question):
        if chunk:
            yield chunk


if __name__ == '__main__':
    # 测试客服问答（混合检索：向量 + BM25 + RRF 融合）
    test_question = "上一届谁是冠军？"
    print(f"用户问题: {test_question}\n")
    for chunk in chat_stream(test_question):
        print(chunk, end="", flush=True)