#!/bin/bash

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 安装Ollama（如果没有）
if ! command -v ollama &> /dev/null; then
    echo "安装Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 下载模型
ollama pull llama3.2
ollama pull nomic-embed-text

echo "环境设置完成！"
echo "请将你的PDF文件放入 data/raw/ 目录"
echo "运行 'python scripts/ingest_data.py' 导入数据"
echo "运行 'python src/main.py' 启动服务"