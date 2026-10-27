import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent #use this when
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from datetime import datetime
import gradio as gr
import sqlite3
from langchain_tavily import TavilySearch
#from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

search_tool = TavilySearch()
conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)
#llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm = ChatOllama(model="qwen2.5:7b")

system_prompt = """
You are a helpful assistant.
Answer all user queries.
Use the search_tool if the user is asking about the latest python version
Use the get_date tool if the user is asking about todays date

"""

agent = create_agent(model=llm,
                     tools=[get_date, search_tool],
                     system_prompt=system_prompt,
                     checkpointer = checkpointer)
#user_query = input("Enter a query: ")


def chat(message, history, thread_id):
    config = {"configurable":{"thread_id":thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},config)
    last_response = response['messages'][-1].content
    print(response)
    return last_response

with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])


demo.launch()


