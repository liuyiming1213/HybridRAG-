# ============================================================
#  2026世界杯 RAG 问答助手 —— 后端容器镜像
#  基于：python:3.12（与本地 venv D:\langchain_env2 同版本）
#  说明：models/（2.6GB 嵌入+重排模型）不打进镜像，运行时用卷挂载
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# 1) 环境变量
#    HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE：模型全部走本地，禁止联网下载
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 2) 系统依赖
#    libgomp1：PyTorch / tokenizers / onnxruntime 多线程运行时必需
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3) PyTorch CPU 版先装（PyPI 默认 torch 是带 CUDA 的大包，用 CPU 源体积小、速度快）
#    注意：requirements.txt 刻意不含 torch/torchaudio/torchvision，就是在这里装
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4) 其余 Python 依赖（利用 Docker 层缓存：仅依赖文件变化时才重装）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 5) 拷贝源码与向量库（models 不拷，走卷挂载）
COPY . .

# 6) 挂载点：本地/宿主的嵌入与重排模型目录映射到容器 /app/models
#    models/ 有 2.6GB，不放镜像层，用 -v 挂载（见下方运行示例）
VOLUME ["/app/models"]

# 7) 对外端口
EXPOSE 8000

# 8) 启动 FastAPI
#    注意：PostgreSQLUtil 目前把数据库 host/password 写死在代码里，
#          部署时数据库必须与容器同网络（Linux 用 --network host 即 host 的 localhost）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]