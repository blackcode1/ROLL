from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

app = FastAPI()

# 模拟数据
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"}
]

# 定义数据模型
class User(BaseModel):
    id: int
    name: str
    email: str

class UserCreate(BaseModel):
    name: str
    email: str

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Server is running"}

# 获取所有用户
@app.get("/users", response_model=List[User])
def get_users():
    return users

# 根据ID获取特定用户
@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# 创建新用户
@app.post("/users", response_model=User)
def create_user(user: UserCreate):
    new_id = max([u["id"] for u in users]) + 1 if users else 1
    new_user = User(id=new_id, name=user.name, email=user.email)
    users.append(new_user)
    return new_user

# 根据ID更新用户
@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user_update: UserCreate):
    for index, user in enumerate(users):
        if user["id"] == user_id:
            updated_user = User(id=user_id, name=user_update.name, email=user_update.email)
            users[index] = updated_user
            return updated_user
    raise HTTPException(status_code=404, detail="User not found")

# 根据ID删除用户
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    global users
    initial_length = len(users)
    users = [u for u in users if u["id"] != user_id]
    if len(users) == initial_length:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"User {user_id} deleted successfully"}

# 主页
@app.get("/")
def home():
    return {
        "message": "Welcome to the simple Python backend service!",
        "endpoints": [
            {"GET": "/health", "description": "Health check"},
            {"GET": "/users", "description": "Get all users"},
            {"GET": "/users/{id}", "description": "Get user by ID"},
            {"POST": "/users", "description": "Create a new user"},
            {"PUT": "/users/{id}", "description": "Update a user"},
            {"DELETE": "/users/{id}", "description": "Delete a user"}
        ]
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)