from pydantic import BaseModel, Field


class ChatHistory(BaseModel):
    question: str = Field(description="问题")
    answer: str = Field(description="答案")
    parentId: int = Field(description="父级ID", default=0)
    userId: int = Field(description="用户ID")
