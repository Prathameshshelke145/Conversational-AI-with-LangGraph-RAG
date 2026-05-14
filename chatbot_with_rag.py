from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os ,requests
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#--------------realted to tool---------------------
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition,ToolNode
import streamlit as st
# --------------------------------Load environment variables
load_dotenv()

#---------------------------------llm
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

print("Google API key loaded successfully!")

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

embed=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Global vector store backup (fallback when st.session_state is not available)
GLOBAL_VECTOR_STORE = None

#----------------------------------------------------------------tool
@tool
def calculator(first_number:float , second_number:float , operation:str)->dict:
    "perform basic arthmatic operations like add, mul ,div ,sub"
    try:
        if operation == "add":
            result=first_number+second_number
        elif operation == "sub":
            result=first_number - second_number
        elif operation == "mul":
            result=first_number * second_number    
        elif operation == "div":
            result=first_number/second_number 
        else :
            return {"error:{operation} error can not perform the operation"}       
        return {f"{first_number} {operation} {second_number} = {result}"}
    except Exception as e:
        return {f"error : {str(e)}"}
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('Alpha_Vantage_API')}"
    r = requests.get(url)
    return r.json()    

@tool
def Rag_tool(query: str) -> str:
    """Fetch relevant context from uploaded PDF"""
    global GLOBAL_VECTOR_STORE

    # Use global vector store directly
    if GLOBAL_VECTOR_STORE is None:
        return "No PDF uploaded yet. Please upload a PDF file first."

    retriever = GLOBAL_VECTOR_STORE.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    docs = retriever.invoke(query)

    return "\n\n".join([doc.page_content for doc in docs])

# GLOBAL_VECTOR_STORE = None
def create_faiss_vector_store(pdf_path: str):
    global GLOBAL_VECTOR_STORE
    
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(doc)

    GLOBAL_VECTOR_STORE = FAISS.from_documents(documents, embed)

    
    return GLOBAL_VECTOR_STORE
    


tools=[calculator,get_stock_price,Rag_tool]
tool_llm=llm.bind_tools(tools)

tool_node=ToolNode(tools)



#--------------------------create daatbase 
conn=sqlite3.connect("chatbot.db",check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)




#----------------------------state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


    
#----------------------------nodes
def chat_node(state:ChatState)->dict:
    messages = state["messages"]

    system = SystemMessage(
        content="""You are a helpful assistant with access to a PDF document.

CRITICAL RULES:
1. If user asks ANYTHING about "PDF", "document", "file", or "what is this", immediately call Rag_tool
2. If user provides a query that could relate to uploaded content, use Rag_tool first
3. Always use Rag_tool to answer questions that might be in the document
4. If Rag_tool returns "No PDF uploaded", inform the user
5. Otherwise, use the retrieved context to answer accurately

Your tools:
- calculator: for math operations
- get_stock_price: for stock information
- Rag_tool: for PDF/document queries (USE THIS FOR DOCUMENT QUESTIONS!)"""
    )

    response = tool_llm.invoke([system] + messages)

    return {"messages": [response]}



#----------------------------graph
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")
chatbot = graph.compile(checkpointer=checkpointer)



#------------------------------extract tread ids 
all_threads=set()
def retriev_all_threads():
    for thread in checkpointer.list(None):
        all_threads.add(thread.config["configurable"]["thread_id"])
    return list(all_threads)