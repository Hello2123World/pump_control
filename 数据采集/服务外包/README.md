# AI健身助手 - 关键词提取应用

这个项目是一个AI健身助手应用，它可以从用户输入的文本中提取关键词，并提供健身相关的建议和计划。

## 功能特点

- 用户可以在聊天界面输入健身相关的问题或描述
- 应用使用DeepSeek API自动提取用户输入的关键词
- 根据用户输入提供健身建议和相关计划
- 支持页面跳转到不同的健身计划页面

## 安装和运行

### 安装依赖

```bash
pip install flask flask-cors openai
```

### 运行应用

1. 启动Flask服务器：

```bash
python app.py
```

2. 在浏览器中访问：

```
http://localhost:5000
```

## 文件结构

- `app.py` - Flask应用服务器
- `ds_api.py` - DeepSeek API交互模块
- `main.html` - 主页面HTML文件
- `dashboard.html` - 仪表盘页面

## API密钥设置

当前应用使用了硬编码的DeepSeek API密钥。在生产环境中，应该将API密钥保存在环境变量或配置文件中，并确保不要将其提交到版本控制系统。

## 依赖项

- Flask - Web应用框架
- Flask-CORS - 处理跨域请求
- OpenAI Python客户端 - 用于调用DeepSeek API

## 使用方法

1. 在输入框中输入健身相关的问题或需求
2. 系统会自动提取关键词并显示
3. 根据输入内容，系统会提供相关建议或跳转到相应的页面
4. 可以输入"打开仪表盘"来查看健身数据仪表盘 