
from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from chat.controller.ChatController import chat_router
from chat.controller.HistoryController import history_router
from users.controller.UsersController import users_router

# 创建FastAPI对象
app = FastAPI()

# 跨域配置（必须在路由注册之前添加）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],#允许哪些前端资源访问后端
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 访问静态资源放行
app.mount("/static", app=StaticFiles(directory="static"), name="static")

# 引入users模块的子路由
app.include_router(router=users_router, prefix="/users", tags=["users"])#tags:标签
# 注册chat模块
app.include_router(router=chat_router, prefix="/chat", tags=["chat"])
#注册history模块
app.include_router(router=history_router, prefix="/history", tags=["history"])

# 运行FastAPI
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=False,
    )