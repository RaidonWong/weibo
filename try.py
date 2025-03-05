import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_URL")
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

response = client.chat.completions.create(
    messages=messages,
    model="/root/autodl-tmp/liuxiaoyou/models/meta-llama/Meta-Llama-3-8B-Instruct",  # 或者尝试完整路径
)

print(response)

