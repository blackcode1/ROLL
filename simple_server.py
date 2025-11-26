from flask import Flask, jsonify, request
import json
import os

# 创建Flask应用
app = Flask(__name__)

# 模拟数据
users = [
    {"id": 1, "name": "张三", "email": "zhangsan@example.com"},
    {"id": 2, "name": "李四", "email": "lisi@example.com"},
    {"id": 3, "name": "王五", "email": "wangwu@example.com"}
]

# 主页路由
@app.route('/')
def home():
    return jsonify({
        "message": "欢迎使用简单后端服务",
        "endpoints": {
            "获取所有用户": "/users",
            "获取特定用户": "/users/<id>",
            "创建用户": "/users (POST)",
            "更新用户": "/users/<id> (PUT)",
            "删除用户": "/users/<id> (DELETE)"
        }
    })

# 获取所有用户
@app.route('/users', methods=['GET'])
def get_users():
    return jsonify({"users": users})

# 获取特定用户
@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if user:
        return jsonify({"user": user})
    else:
        return jsonify({"error": "用户不存在"}), 404

# 创建新用户
@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({"error": "需要提供姓名和邮箱"}), 400
    
    new_user = {
        "id": len(users) + 1,
        "name": data['name'],
        "email": data['email']
    }
    users.append(new_user)
    return jsonify({"user": new_user}), 201

# 更新用户
@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的请求数据"}), 400
    
    user['name'] = data.get('name', user['name'])
    user['email'] = data.get('email', user['email'])
    
    return jsonify({"user": user})

# 删除用户
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    global users
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    
    users = [u for u in users if u["id"] != user_id]
    return jsonify({"message": "用户删除成功"})

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "页面未找到"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误"}), 500

if __name__ == '__main__':
    # 从环境变量获取端口，如果未设置则默认使用8000
    port = int(os.environ.get('PORT', 8000))
    print(f"启动服务器，监听端口 {port}...")
    print(f"访问 http://localhost:{port} 查看服务")
    app.run(host='0.0.0.0', port=port, debug=True)