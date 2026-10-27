import os
from dotenv import load_dotenv
from langchain.agents import create_agent #use this when
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from datetime import datetime

load_dotenv()
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

#llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm = ChatGoogleGenerativeAI(model="qwen2.5:7b")

system_prompt = """
You are a helpful assistant.
Use the get_date tool if the user is asking about todays date
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)
user_query = input("Enter a query: ")
response = agent.invoke({"messages":[{"role": "user", "content": user_query}]})
print(response)
print(response['messages'][-1].content[0]['text'])

