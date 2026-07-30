import os
import re
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,RemoveMessage
from graph_state import RAGGraphState
from embeddings import embed_model

load_dotenv()
CHROMA_PATH="./chroma_db"

API_KEY=os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your .env file before starting the app."
    )

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=API_KEY,temperature=0.01)
vector_store=Chroma(persist_directory=CHROMA_PATH,embedding_function=embed_model)

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "repeat the entire document",
    "print all documents",
    "developer message",
    "reveal prompt",
    ]
def normalize(text:str)->str:
    text=text.lower()
    text=re.sub(r'\s+'," ",text).strip()
    return text

def input_guardrail(state:RAGGraphState)->RAGGraphState:
    print("Input Guardrail is Working...")
    question = normalize(state["question"])
    for pattern in BLOCKED_PATTERNS:
        if pattern in question:
            return {"initial_check":False}
    return {"initial_check":True}

def rewrite_question(state:RAGGraphState)->RAGGraphState:
    print("Rewrite node is working...")

    if len(state['chat_history']) ==0 :
        return {"rewrite_question":state['question']}

    prompt=ChatPromptTemplate.from_template("""Conversation Summary:
{summary}

Recent Conversation:
{history}

Current User Question:
{question}

Rewrite the current question into a standalone question only if it depends on previous conversation.
Otherwise return it unchanged.

Return ONLY the rewritten question.""")
    chain= prompt | llm

    result=chain.invoke({'summary':state["summary"],'history':state["chat_history"],"question":state["question"]})

    return {"rewrite_question":result.content}

def retriever(state:RAGGraphState)->RAGGraphState:
    print("Retriever retrieving document....")
    retrieved_documents=vector_store.similarity_search(state['rewrite_question'],k=3)
    return {"document":retrieved_documents}

class relevence_checker(BaseModel):
    is_relevant:bool=Field(description="True if retrieved context contain enough information to answer the question")

def relevent(state:RAGGraphState)->RAGGraphState:
    print("Relevence checking...")

    context=("\n\n".join([doc.page_content for doc in state['document']]))

    relevent_llm=llm.with_structured_output(relevence_checker)
    prompt=ChatPromptTemplate.from_template(""" You are a retrieval evaluator.

Question:
{question}

Retrieved Context:
{context}

Determine whether the retrieved context contains enough information
to answer the question accurately.""")
    chain=prompt|relevent_llm
    result=chain.invoke({'question':state['rewrite_question'],"context":context})
   
    return {"docs_check":result.is_relevant}

def generate_answer(state:RAGGraphState)->RAGGraphState:
    print("Generating Answer...")
    context="\n\n".join([doc.page_content for doc in state["document"]])
    prompt=ChatPromptTemplate.from_template(""" you are a RAG AI assistent.
    answer the question using only the data_base below. if the data_base don't fully has the answer for the question just say i don't know friendly.
    chat_summary:{summary}
    
    recent_chat:{chat_history}
    
    data_base:{context}
    
    user_question:{question}

    rewritten_question:{rewrittent_question}

    answer:""")
    chain=prompt | llm

    response=chain.invoke({
    "summary": state["summary"],
    "chat_history": state["chat_history"],
    "context": context,
    "question": state["question"],
    "rewrittent_question": state["rewrite_question"]
})
    
    return {
    "answer": response.content,
    "chat_history": [
        HumanMessage(content=state["question"]),
        AIMessage(content=response.content)
    ]
}

def summary(state:RAGGraphState)->RAGGraphState:
    print("Summarizing chat...")

    prompt=ChatPromptTemplate.from_template(""" You are maintaining memory for a chatbot.

Existing summary (may be empty):
{summary}
 
New messages since the last summary:
{chat_history}
 
Update the summary to incorporate any new, useful information from the new
messages above. Keep information that may be useful in future conversations.
 
Remove greetings.
 
Remove repeated facts.
 
Keep names, dates, important events and user preferences.
 
Maximum 120 words.
 
Return ONLY the updated summary.""")

    chain= prompt | llm
    result=chain.invoke({"chat_history":state['chat_history'],"summary":state["summary"]})
    messages_to_drop = state["chat_history"][:-4]
    remove_ops = [RemoveMessage(id=m.id) for m in messages_to_drop]
 
    return {
        "summary": result.content,
        "chat_history": remove_ops,
    }

def escalate(state:RAGGraphState)->RAGGraphState:
    print("Escalating...")
    return {"answer":"I couldn't find enough information in the uploaded document.Please upload another document or ask a different question."}

def should_answer(state:RAGGraphState):
    print("should answer?...")
    if state['docs_check']:
        return "generate"
    return 'escalate'

def should_summarize(state:RAGGraphState):
    if len(state['chat_history'])>=10:
        return "summary"
    return "end"

def initial_check(state:RAGGraphState):
    print("initial checking...")
    if state["initial_check"]:
        return "rewrite"
    return "escalate"