from langgraph.graph import StateGraph,END
from graph_state import RAGGraphState
from nodes import (
    input_guardrail,
    rewrite_question,
    retriever,
    relevent,
    generate_answer,
    summary,
    should_answer,
    should_summarize,
    initial_check,
    escalate
)

graph=StateGraph(RAGGraphState)

graph.add_node("input_guardrail",input_guardrail)
graph.add_node("rewrite_question",rewrite_question)
graph.add_node("retriever",retriever)
graph.add_node("relevent",relevent)
graph.add_node("generate_answer",generate_answer)
graph.add_node("summary",summary)
graph.add_node("escalate",escalate)

graph.set_entry_point("input_guardrail")
graph.add_conditional_edges("input_guardrail",initial_check,{"rewrite":"rewrite_question","escalate":"escalate"})
graph.add_edge("rewrite_question","retriever")
graph.add_edge("retriever","relevent")
graph.add_conditional_edges("relevent",should_answer,{"generate":"generate_answer","escalate":"escalate"})
graph.add_conditional_edges("generate_answer",should_summarize,{"summary":"summary","end":END})
graph.add_edge("summary",END)
graph.add_edge("escalate",END)

app=graph.compile()

initial_state={"question":"",
               "initial_check":False,
               "rewrite_question":"",
               "document":[],
               "docs_check":False,
               "answer":"",
               "chat_history":[],
               "summary":""
               }

# initial_state["question"]="how can i request a refund?"
# response=app.invoke(initial_state)

# initial_state=response

# print("Answer :",response["answer"])