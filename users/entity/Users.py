# Users 类
from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass
class Users(BaseModel):
    username: str = Field(description="用户名")
    password: str = Field(description="密码")