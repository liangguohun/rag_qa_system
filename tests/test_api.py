import requests

try:
    response = requests.post(
        "http://localhost:8000/ask",
        # json={"question": "总结一下这个文档的核心观点"},
        json={"question": "vue3 reactor 使用细节及详解"},
        # timeout=30
    )
    print("状态码:", response.status_code)
    print("返回内容:", response.json())
except Exception as e:
    print("错误 →", e)