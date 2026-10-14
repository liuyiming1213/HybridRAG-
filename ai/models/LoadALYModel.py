#加载阿里云百炼平台模型---千问模型
import os

from langchain_classic.agents import agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def load_model():
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://ws-10wk3k5xp7l1fl01.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-flash",
        streaming=True,
    )

