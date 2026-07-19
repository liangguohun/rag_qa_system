from openai import OpenAI

# 根据你的Key来源，选择对应的base_url
client = OpenAI(
    api_key="dacc77c57e2e4e5ab307dd010c52d3c9.SgfGeAcMRfBAG1dT",
    base_url="https://open.bigmodel.cn/api/paas/v4/"  # 或 https://api.z.ai/api/paas/v4
)

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)