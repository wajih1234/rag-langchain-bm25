import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=500,
)

if __name__ == "__main__":
    print(llm.invoke(" who is  cr7.").content)