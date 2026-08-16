import os

import anthropic

# 初始化客户端，如果您已经通过环境变量 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_API_KEY` 
# 设置了 API Key 和 base URL，可以省略 `api_key` 和 `base_url` 参数。
client = anthropic.Anthropic(
    # 重写 header
    default_headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['ANTHROPIC_API_KEY']}",
    }
)

message = client.messages.create(
    model="moonshotai/kimi-k2-instruct",
    max_tokens=1000,
    temperature=1,
    system=[
        {
            "type": "text",
            "text": "你是 JieKou AI AI 助手，你会以诚实专业的态度帮助用户，用中文回答问题。"
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "你是谁?"
                }
            ]
        }
    ]
)

print(message.content)
