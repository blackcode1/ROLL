from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import datetime

# 创建FastAPI应用实例
app = FastAPI(title="简单Python后端服务", version="1.0.0")

# 定义数据模型
class Item(BaseModel):
    id: int
    name: str
    value: int

class SampleData(BaseModel):
    items: List[Item]
    total_count: int
    description: str

class PostDataRequest(BaseModel):
    data: dict

class PostDataResponse(BaseModel):
    message: str
    received_data: dict
    status: str

# 设置一个简单的路由，返回欢迎信息
@app.get("/")
async def home():
    return {
        "message": "欢迎来到简单的Python后端服务！",
        "status": "running",
        "endpoints": {
            "/": "此主页",
            "/api/status": "服务状态",
            "/api/hello/{name}": "问候API",
            "/api/data": "示例数据API (GET/POST)"
        }
    }

# 服务状态API
@app.get("/api/status")
async def status():
    return {
        "status": "success",
        "service": "simple_backend",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

# 简单的问候API
@app.get("/api/hello/{name}")
async def hello(name: str):
    return {
        "message": f"你好，{name}！欢迎使用此服务！",
        "received_name": name
    }

# 示例数据API
@app.get("/api/data", response_model=SampleData)
async def get_data():
    # 返回示例数据
    sample_data = {
        "items": [
            {"id": 1, "name": "示例项目1", "value": 100},
            {"id": 2, "name": "示例项目2", "value": 200},
            {"id": 3, "name": "示例项目3", "value": 300}
        ],
        "total_count": 3,
        "description": "这是一个示例数据API"
    }
    return sample_data

# 示例数据API - POST请求
@app.post("/api/data", response_model=PostDataResponse)
async def post_data(request: PostDataRequest):
    return {
        "message": "数据接收成功",
        "received_data": request.data,
        "status": "success"
    }

# 错误处理
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "请求的资源不存在",
            "status": 404
        }
    )

@app.exception_handler(500)
async def internal_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "status": 500
        }
    )

if __name__ == '__main__':
    # 获取端口，优先使用环境变量，否则默认使用9000
    port = int(os.environ.get('PORT', 9000))
    print(f"启动简单的Python后端服务...")
    print(f"服务将在 http://localhost:{port} 上运行")
    uvicorn.run(app, host='0.0.0.0', port=port)