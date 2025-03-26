from flask import Flask, request, jsonify
from flask_cors import CORS  # 处理跨域请求
import ds_api  # 导入现有的ds_api模块

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # 启用跨域支持

@app.route('/')
def index():
    return app.send_static_file('main.html')

@app.route('/dashboard.html')
def dashboard():
    return app.send_static_file('dashboard.html')

@app.route('/muscle-building.html')
def muscle_building():
    return app.send_static_file('muscle-building.html')

@app.route('/fat-loss.html')
def fat_loss():
    return app.send_static_file('fat-loss.html')

@app.route('/body-sculpting.html')
def body_sculpting():
    return app.send_static_file('body-sculpting.html')

@app.route('/rehabilitation.html')
def rehabilitation():
    return app.send_static_file('rehabilitation.html')

@app.route('/api/keywords', methods=['POST'])
def get_keywords():
    # 获取用户输入的文本
    user_input = request.json.get('text', '')
    
    if not user_input:
        return jsonify({"error": "没有提供文本内容"}), 400
    
    # 调用DeepSeek API处理文本
    keywords = ds_api.process_text(user_input)
    
    return jsonify({"keywords": keywords})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 