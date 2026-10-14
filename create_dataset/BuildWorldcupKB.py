"""
世界杯知识库构建 + 测试脚本
============================

功能：
  1. 读取 5 个数据集（CSV格式），将每条数据转为一句自然语言文本
  2. 调用 BGE 嵌入模型将文本转为向量
  3. 存入 ChromaDB 向量数据库，供问答检索

核心思路：
  数据集是英文的（队名、阶段名等），但用户会用中文提问。
  所以在转文本时，额外给英文队名附上中文翻译，如：
    "Argentina(阿根廷) vs Brazil(巴西)"
  这样用户搜"阿根廷"时，向量也能匹配到这条文档。

数据集来源（都在 datasets/ 目录下）：
  archive2/schedule_2026.csv        → 2026世界杯赛程（72场比赛）
  archive2/world_cup.csv            → 历届世界杯摘要（22届）
  archive2/fifa_ranking_2026-06-08.csv → FIFA世界排名（211队）
  archive2/matches_1930_2022.csv    → 历史世界杯关键比赛
  archive4/results.csv              → 所有国际比赛（仅取世界杯的）
"""

import os
import torch
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ai.models import LoadALYModel
import glob
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- 离线模式：所有模型从本地加载，不走外网 ----
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---- 路径配置 ----
# path_base = 当前文件往上两级（D:/stu_fastapi）
path_base = os.path.dirname(os.path.dirname(__file__))
embedding_model_name = os.path.abspath(os.path.join(path_base, 'models', 'embedding_model'))     # BGE嵌入模型
reranker_model_name  = os.path.join(path_base, 'models', 'bge-reranker-large_v1')               # BGE重排序模型
datasets_path        = os.path.join(path_base, 'datasets')                                       # 数据集根目录
vector_db_path       = os.path.join(path_base, 'vector_db')                                      # 向量库存放位置
vector_db_name       = 'worldcup_kb'                                                             # 集合名


#将国家名中英文进行转换
TEAM_CN = {
    # 参赛球队（2026世界杯的48队 + 历史强队）
    "Mexico": "墨西哥", "South Africa": "南非", "Argentina": "阿根廷",
    "Algeria": "阿尔及利亚", "Jordan": "约旦", "Austria": "奥地利",
    "Australia": "澳大利亚", "United States": "美国", "Paraguay": "巴拉圭",
    "Brazil": "巴西", "Portugal": "葡萄牙", "Colombia": "哥伦比亚",
    "Uruguay": "乌拉圭", "Spain": "西班牙", "Germany": "德国",
    "France": "法国", "England": "英格兰", "Italy": "意大利",
    "Netherlands": "荷兰", "Belgium": "比利时", "Croatia": "克罗地亚",
    "Morocco": "摩洛哥", "Japan": "日本", "Korea Republic": "韩国",
    "South Korea": "韩国", "Senegal": "塞内加尔", "Iran": "伊朗",
    "Saudi Arabia": "沙特阿拉伯", "Canada": "加拿大", "Denmark": "丹麦",
    "Switzerland": "瑞士", "Serbia": "塞尔维亚", "Poland": "波兰",
    "Tunisia": "突尼斯", "Ghana": "加纳", "Cameroon": "喀麦隆",
    "Ecuador": "厄瓜多尔", "Peru": "秘鲁", "Chile": "智利",
    "Norway": "挪威", "Sweden": "瑞典", "Wales": "威尔士",
    "Costa Rica": "哥斯达黎加", "Egypt": "埃及", "Nigeria": "尼日利亚",
    "Qatar": "卡塔尔", "New Zealand": "新西兰", "Scotland": "苏格兰",
    "Ukraine": "乌克兰", "Greece": "希腊", "Czech Republic": "捷克共和国",
    "Czechia": "捷克", "Slovakia": "斯洛伐克", "Hungary": "匈牙利",
    "Romania": "罗马尼亚", "Venezuela": "委内瑞拉", "Bolivia": "玻利维亚",
    "Panama": "巴拿马", "Russia": "俄罗斯", "Turkey": "土耳其",
    "Türkiye": "土耳其", "Cote d'Ivoire": "科特迪瓦",
    "Côte d'Ivoire": "科特迪瓦", "China": "中国",
}

ROUND_CN = {
    # 比赛阶段 → 中文（CSV和JSON里用的不同格式，都覆盖）
    "Group stage": "小组赛", "Round of 32": "32强赛", "Round of 16": "16强赛",
    "Quarter-final": "四分之一决赛", "Semi-final": "半决赛",
    "Third-place match": "三四名决赛", "Final": "决赛",
    "group": "小组赛", "round_of_32": "32强", "round_of_16": "16强",
    "quarter": "八强", "semi": "半决赛", "final": "决赛",
}

def team_cn(name):
    """根据英文名查中文名，找不到就当英文名用"""
    return TEAM_CN.get(name, name)

def round_cn(name):
    """根据英文阶段名查中文，找不到就当英文用"""
    return ROUND_CN.get(name, name)



def convert_schedule_2026():
    """
    将 schedule_2026.csv 每一行转成一个 Document。

    每行代表一场比赛。CSV 从两队视角各记录一次（如 Mexico vs Argentina 会有两行），
    这里保留全部是为了让"找某队的所有比赛"时都能命中。
    """
    docs = []
    path = os.path.join(datasets_path, 'archive2', 'schedule_2026.csv')

    if not os.path.exists(path):
        print("  [WARN] schedule_2026.csv 未找到")
        return docs

    df = pd.read_csv(path)                         # pandas 读 CSV 变成 DataFrame（表格）
    for _, row in df.iterrows():                   # 逐行遍历，row 是一行的数据
        # 1. 取出需要的字段
        rnd      = str(row.get('Round', ''))       # 阶段，如 "Group stage"
        date_raw = str(row.get('Date', ''))        # 日期，如 "2026-06-11"
        time_raw = str(row.get('Time', ''))        # 时间，如 "13:00 (22:00)"
        home     = str(row.get('home_team', ''))   # 主队
        away     = str(row.get('away_team', ''))   # 客队

        # 2. 翻译成中文
        home_cn = team_cn(home)
        away_cn = team_cn(away)
        rnd_cn  = round_cn(rnd)

        # 3. 日期格式从 "2026-06-11" → "2026年6月11日"
        parts = date_raw.split('-')
        if len(parts) == 3:
            date_cn = f"{parts[0]}年{int(parts[1])}月{int(parts[2])}日"
        else:
            date_cn = date_raw

        # 4. 拼接队名显示（同时含英文和中文）
        home_display = f"{home}({home_cn})" if home_cn != home else home
        away_display = f"{away}({away_cn})" if away_cn != away else away

        # 5. 拼成一句自然语言文本（核心：这条文本会被向量化，用于语义检索）
        text = (f"2026世界杯，{rnd_cn}({rnd})，{date_cn}，"
                f"{home_display} vs {away_display}，UTC时间：{time_raw}")

        # 6. 包装为 LangChain Document 对象
        docs.append(Document(
            page_content=text,                     # 正文（会被向量化）
            metadata={"source": "schedule_2026"}   # 元数据（来源标记）
        ))

    print(f"  [schedule_2026.csv] 转换 {len(docs)} 条")
    return docs



def convert_world_cup_history():
    """
    历届世界杯冠军、亚军、金靴等信息。
    用户问"2018年世界杯谁夺冠了？"就能从这里查到。
    """
    docs = []
    path = os.path.join(datasets_path, 'archive2', 'world_cup.csv')

    if not os.path.exists(path):
        print("  [WARN] world_cup.csv 未找到")
        return docs

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        year       = int(row['Year'])
        host       = str(row['Host']).strip()
        champion   = str(row['Champion']).strip()
        runner_up  = str(row['Runner-Up']).strip()
        top_scorer = str(row.get('TopScorrer'  , '')).strip()
        teams      = int(row.get('Teams', 0))

        # 查中文名
        champ_cn  = team_cn(champion)
        runner_cn = team_cn(runner_up)

        text = (f"{year}年{host}世界杯，参赛队伍{teams}支，"
                f"冠军：{champion}({champ_cn})，亚军：{runner_up}({runner_cn})，"
                f"金靴奖得主：{top_scorer}")

        docs.append(Document(page_content=text, metadata={"source": "world_cup_history", "year": year}))

    print(f"  [world_cup.csv] 转换 {len(docs)} 条")
    return docs


def convert_fifa_ranking():
    """
    每条排名一条文档。
    用户问"巴西排名第几？"或"FIFA排名前十有哪些？"时能检索到。
    """
    docs = []
    path = os.path.join(datasets_path, 'archive2', 'fifa_ranking_2026-06-08.csv')

    if not os.path.exists(path):
        print("  [WARN] fifa_ranking 未找到")
        return docs

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        team    = row['team']
        code    = row['team_code']
        rank    = int(row['rank'])
        points  = round(float(row['points']), 2)
        conf    = row['association']
        tc      = team_cn(team)

        text = (f"FIFA世界排名(2026年6月)，第{rank}名：{team}({tc})，"
                f"代码{code}，积分{points}分，所属足联{conf}")

        docs.append(Document(page_content=text, metadata={"source": "fifa_ranking"}))

    print(f"  [fifa_ranking] 转换 {len(docs)} 条")
    return docs


def convert_historical_matches():
    """
    历史世界杯比赛。只取决赛、半决赛等关键轮次，避免数据量爆炸。
    """
    docs = []
    path = os.path.join(datasets_path, 'archive2', 'matches_1930_2022.csv')

    if not os.path.exists(path):
        print("  [WARN] matches_1930_2022.csv 未找到")
        return docs

    df = pd.read_csv(path)

    # 只保留关键轮次（用 pandas 的 isin 过滤）
    key_rounds = ['Final', 'Semi-final', 'Quarter-final', 'Third-place match']
    key_df = df[df['Round'].isin(key_rounds)]

    for _, row in key_df.iterrows():
        year    = int(row['Year'])
        rnd     = str(row['Round'])
        home    = str(row['home_team'])
        away    = str(row['away_team'])
        h_score = int(row['home_score'])
        a_score = int(row['away_score'])
        host    = str(row.get('Host', ''))
        date    = str(row.get('Date', ''))
        venue   = str(row.get('Venue', '')).split(',')[0]  # 只取场馆名，后面的城市/国家不要

        home_cn = team_cn(home)
        away_cn = team_cn(away)
        rnd_cn  = round_cn(rnd)

        text = (f"{year}年{host}世界杯，{rnd_cn}({rnd})，{date}，"
                f"{home}({home_cn}) {h_score}-{a_score} {away}({away_cn})，场地：{venue}")

        docs.append(Document(page_content=text, metadata={"source": "matches_history", "year": year}))

    print(f"  [matches_1930_2022.csv] 转换 {len(docs)} 条（关键比赛）")
    return docs



def convert_results_wc_only():
    """
    从 results.csv 中只提取世界杯赛事。
    """
    docs = []
    path = os.path.join(datasets_path, 'archive4', 'results.csv')

    if not os.path.exists(path):
        print("  [WARN] results.csv 未找到")
        return docs

    df = pd.read_csv(path)

    # 筛选：只取 tournament 列里包含 "FIFA World Cup" 的行
    # na=False 表示空值不参与匹配（否则报错）
    # case=False 表示不区分大小写
    wc_df = df[df['tournament'].str.contains('FIFA World Cup', na=False, case=False)]

    for _, row in wc_df.iterrows():
        date       = str(row['date'])
        home       = str(row['home_team'])
        away       = str(row['away_team'])
        # NaN 处理：CSV 中有些比分是空的（如还没比赛的未来赛程），用 0 代替
        h_score    = int(row['home_score']) if not pd.isna(row['home_score']) else 0
        a_score    = int(row['away_score']) if not pd.isna(row['away_score']) else 0
        tournament = str(row['tournament'])
        city       = str(row.get('city', ''))

        home_cn = team_cn(home)
        away_cn = team_cn(away)

        text = (f"世界杯比赛，{date}，{home}({home_cn}) {h_score}-{a_score} "
                f"{away}({away_cn})，赛事：{tournament}，举办城市：{city}")

        docs.append(Document(page_content=text, metadata={"source": "results_wc"}))

    print(f"  [results.csv] 转换 {len(docs)} 条（仅世界杯比赛）")
    return docs



# ================================================================
#  PDF 非结构化文本接入（扩展数据源）
#  功能：把 datasets/pdf/ 目录下的 PDF 文档用 PyMuPDF 抽取文本，
#       再经 RecursiveCharacterTextSplitter 按标题/段落语义层级切块，
#       输出与 CSV 数据源一致的 Document 列表，共用同一嵌入/检索链路。
#
#  为什么需要 PDF 切块：
#    - CSV 是结构化表格，逐行即可成为独立 Document
#    - PDF 是整篇长文，直接整篇入库会让每个语义单元过大，
#      检索时命中整篇、带入 Prompt 的噪声 Token 多、召回不精准
#    - 按标题/段落切块后，检索返回的粒度 = 语义块，召回更准、成本更低
# ================================================================
def convert_pdf():

    pdf_dir = os.path.join(datasets_path, 'pdf')
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, '*.pdf')))
    if not pdf_files:
        print("  [WARN] datasets/pdf/ 下没有 PDF 文档")
        return []

    # 语义层级切块器：优先按段落(空行)切，其次按换行/句号等标点兜底
    # chunk_size = 单个 chunk 目标字符数，chunk_overlap = 相邻块重叠窗口
    # 重叠窗口用于保住跨块上下文的衔接，避免一句话被切断后两端各自不完整
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=False,
    )

    docs = []
    for pdf in pdf_files:
        with pymupdf.open(pdf) as doc:
            # 整篇文本（用于判断是否为空/扫描件）
            full_text = "".join(page.get_text("text") for page in doc)
            if not full_text.strip():
                print(f"  [WARN] {os.path.basename(pdf)} 无文本，可能为扫描件")
                continue

            # 逐页抽取+切块：以"页"为粗粒度语义单元（一页通常对应一个章节/小节），
            # 再对过长的页用 RecursiveCharacterTextSplitter 细化切块。
            # 相比"整篇一次切"，逐页切能保住章节结构，检索命中粒度更精准。
            chunk_idx = 0
            for page_no, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                if not page_text.strip():
                    continue
                page_chunks = splitter.split_text(page_text)
                for chunk in page_chunks:
                    if not chunk.strip():
                        continue
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": "pdf",                       # 检索来源标记
                            "file": os.path.basename(pdf),         # 来自哪个PDF
                            "page": page_no,                       # 所在页码
                            "chunk": chunk_idx,                    # 全局块序号
                        },
                    ))
                    chunk_idx += 1
        print(f"  [convert_pdf] {os.path.basename(pdf)} -> {chunk_idx} 个chunk")
    return docs


# 增量接入：把 datasets/pdf 切块后写入已有的 ChromaDB 集合（不重建既有知识库）
# 适用于在不触碰原有约2万文档的前提下，追加 PDF 长文数据源
def add_pdf_to_kb():
    from chat.util import ChromaUtil
    docs = convert_pdf()
    if not docs:
        print("[WARN] 没有 PDF 文档可入库")
        return
    db = ChromaUtil.get_kb_conn()
    ids = [f"pdf-{d.metadata['file']}-{d.metadata['chunk']}" for d in docs]
    db.add_documents(documents=docs, ids=ids)
    print(f"[OK] 已增量接入 {len(docs)} 个 PDF chunk 到集合 {vector_db_name}")


def build_worldcup_kb():
    """
    构建世界杯知识库的主函数。

    步骤：
    1. 调用上面 6 个转换函数（5个CSV + 1个PDF），得到 Document 对象列表
    2. 加载 BGE 嵌入模型
    3. 调用 Chroma.from_documents() 自动对每条文本做向量化并存入本地
    """
    # 收集所有文档
    all_docs = []
    all_docs.extend(convert_schedule_2026())       # 2026赛程
    all_docs.extend(convert_world_cup_history())   # 历届摘要
    all_docs.extend(convert_fifa_ranking())        # FIFA排名
    all_docs.extend(convert_historical_matches())  # 历史比赛
    all_docs.extend(convert_results_wc_only())     # 世界杯结果
    all_docs.extend(convert_pdf())                 # PDF非结构化长文

    print(f"\n总计 {len(all_docs)} 条文档待入库")

    # 加载 BGE 嵌入模型
    # "语义相近"的文本在向量空间中距离也近
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},                        # CPU 推理
        encode_kwargs={"normalize_embeddings": True,           # 向量归一化
                       "batch_size": 32},                      # 每批处理 32 条
    )

    # 向量化 + 存入 ChromaDB
    # from_documents 内部会：
    #   1. 遍历每个 Document，取 page_content
    #   2. 调用 embedding_model 将文本转成向量
    #   3. 将 文本原文 + 向量 + metadata 一起存入 ChromaDB
    Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        persist_directory=vector_db_path,                      # 数据存硬盘的路径
        collection_name=vector_db_name,                        # 集合名（类似 MySQL 的表名）
        collection_metadata={"hnsw:space": "cosine"},          # 用余弦相似度做检索
    )

    print(f"[OK] 世界杯知识库入库完成!! 共 {len(all_docs)} 条文档")
    print(f"   存放位置: {vector_db_path}")
    print(f"   集合名: {vector_db_name}")


def test_worldcup_kb():
    """测试知识库的检索质量和大模型回答"""

    # ---- 1. 连接知识库 ----
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory=vector_db_path,
        collection_name=vector_db_name,
        embedding_function=embedding_model,
    )

    # ---- 2. 创建检索器（每次检索返回 Top-10 最相关文档） ----
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # ---- 3. 测试问题 ----
    test_question = "阿根廷队伍世界排名？"
    print(f"\n测试问题: {test_question}")
    print(f"检索到 {len(retriever.invoke(test_question))} 篇相关文档")

    # ---- 4. 加载大模型（阿里云千问） ----
    llm = LoadALYModel.load_model()

    # ---- 5. Prompt 模板 ----
    template = """
        你是一个世界杯球队分析问答助手。

        【推理要求】
        1.【提取信息】从上下文中提取与问题相关的关键数据
        2.【组织归纳】将零散数据按时间或主题归纳整理，去掉重复
        3.【分析判断】对归纳后的数据进行对比、趋势分析
        4.【最终结论】用简洁的中文给出完整的分析结论

        球队名同时给出中英文，日期用中文格式。

        【知识库上下文】
        {context}

        【用户问题】
        {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # ---- 6. 加载重排序模型（BGE-Reranker） ----
    print("[LOADING] 加载Reranker...")
    tok = AutoTokenizer.from_pretrained(reranker_model_name, local_files_only=True)
    rm  = AutoModelForSequenceClassification.from_pretrained(
        reranker_model_name, local_files_only=True, ignore_mismatched_sizes=True
    )
    rm.eval()

    # ---- 7. 打印粗检索结果（调试用） ----
    def show_docs(docs):
        """打印向量粗检索（Top-10）的结果"""
        print(f"\n{'='*30} 粗检索结果 (Top-10) {'='*30}")
        for i, d in enumerate(docs, 1):
            score = d.metadata.get("score", "N/A")
            print(f"\n  [{i}] {d.page_content}")
            print(f"      来源: {d.metadata.get('source', 'unknown')}")
        return docs  # 必须返回，否则下游拿不到数据

    # ---- 8. 重排序函数（Cross-Encoder 精排） ----
    def rerank(inputs):
        """
        Cross-Encoder 精排：对粗检索的文档逐对计算相关性分数，
        去重后按分降序排列，取 Top-3 送入 LLM。
        """
        docs = inputs["context"]   # 粗检索返回的文档列表
        q    = inputs["question"]  # 用户问题

        # 去重：有些文档内容几乎一样（同场比赛从两队视角各记录一次）
        # 取前200字做"指纹"，相同指纹的认为重复，只保留第一个
        uniq, seen = [], set()
        for d in docs:
            fp = d.page_content[:200]
            if fp not in seen:
                seen.add(fp)
                uniq.append(d)
        print(f"\n  去重后剩余 {len(uniq)} 篇候选文档")
        docs = uniq

        if not docs:
            print("  无文档可重排序")
            return {"context": "", "question": q}

        # 构造 [问题, 文档] 对 → Cross-Encoder → 计算相关性分数
        print(f"  正在对 {len(docs)} 篇文档进行重排序(Coss-Encoder)...")
        pairs = [[q, d.page_content] for d in docs]
        with torch.no_grad():
            t = tok(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = rm(**t, return_dict=True).logits.view(-1).float().tolist()

        # 把分数写入 metadata，按分降序排列，取 Top-3
        for d, s in zip(docs, scores):
            d.metadata["score"] = round(float(s), 4)
        docs_sorted = sorted(docs, key=lambda d: d.metadata["score"], reverse=True)
        top3 = docs_sorted[:3]

        # ---- 打印重排序后的 Top-3 ----
        print(f"\n{'='*30} 重排序结果 (Top-3) {'='*30}")
        for i, d in enumerate(top3, 1):
            print(f"\n  [{i}] 相关度分数: {d.metadata['score']}")
            print(f"      {d.page_content}")
            print(f"      来源: {d.metadata.get('source', 'unknown')}")

        # 将 3 篇文档拼接成一个 context 字符串
        ctx = "\n\n".join(f"[第{i+1}来源] {d.page_content}" for i, d in enumerate(top3))
        return {"context": ctx, "question": q}

    # ---- 9. 组装问答链 ----
    # 链的数据流：
    #   问题字符串
    #     → {context: 检索→打印→去重→重排序→拼接,
    #        question: 原样透传}
    #     → Prompt模板填充
    #     → LLM推理
    #     → 字符串输出
    chain = (
        RunnableParallel({
            "context":  retriever | RunnableLambda(show_docs),
            "question": RunnablePassthrough(),
        })
        | RunnableLambda(rerank)
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "=" * 20 + " 大模型回答 " + "=" * 20)
    print(chain.invoke(test_question))


# ================================================================
#  主入口
# ================================================================
if __name__ == '__main__':
    # 第一步：先执行一次构建（成功后注释掉，避免每次运行都重建）
    #build_worldcup_kb()

    # 第二步：测试
    test_worldcup_kb()
