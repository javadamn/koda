from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

response = llm.invoke("Say hello from the correct model.")
print(response.content)
