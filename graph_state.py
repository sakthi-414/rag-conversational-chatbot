from typing import TypedDict,List,Annotated
from langgraph.graph import StateGraph,END
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RAGGraphState(TypedDict):
    question:str
    initial_check:bool
    rewrite_question:str
    document:List[Document]
    docs_check:bool
    answer:str
    chat_history:Annotated[List[BaseMessage], add_messages]
    summary:str