\# Conversational RAG Chatbot



A conversational RAG (Retrieval-Augmented Generation) chatbot built with \*\*LangGraph\*\*, \*\*Gemini\*\*, and \*\*Chroma\*\*. Unlike a naive `retrieve → generate` pipeline, this project models the conversation as a state graph with input guardrails, query rewriting, retrieval-relevance grading, escalation, and rolling conversation summarization for long-term memory management.



\## Why a graph instead of a simple chain?



A single-pass RAG chain breaks down in real conversations:

\- Follow-up questions ("what about the second one?") don't make sense to a retriever without context from earlier turns.

\- Not every retrieval is good enough to answer from — a naive pipeline will confidently hallucinate anyway.

\- Long conversations blow up the token budget if you keep sending full history to the LLM every turn.



This project addresses each of those with a dedicated graph node instead of hoping one prompt handles everything.



\## Architecture



input\_guardrail → rewrite\_question → retriever → relevent

├── (relevant) → generate\_answer → \[summary if history is long] → END

└── (not relevant) → escalate → END





\- \*\*`input\_guardrail`\*\* — blocks basic prompt-injection patterns before anything else runs.

\- \*\*`rewrite\_question`\*\* — rewrites follow-up questions into standalone queries using chat summary + recent history, so retrieval doesn't miss context-dependent questions.

\- \*\*`retriever`\*\* — similarity search against a Chroma vector store built from an uploaded PDF.

\- \*\*`relevent`\*\* — an LLM-as-judge node (structured output) that checks whether retrieved chunks actually contain enough info to answer, before generating anything.

\- \*\*`generate\_answer`\*\* — answers strictly from retrieved context; explicitly told to say "I don't know" rather than guess.

\- \*\*`summary`\*\* — once chat history passes a threshold, older messages are summarized and removed from the live context window (see \[Memory management](#memory-management) below).

\- \*\*`escalate`\*\* — a safe fallback when the guardrail blocks a request or retrieval isn't good enough to answer from.



\## Memory management



Chat history is stored using LangGraph's `add\_messages` reducer, which \*\*merges by message ID\*\* rather than overwriting the list. That means trimming history isn't as simple as returning a shorter list — old messages have to be explicitly removed with `RemoveMessage`, or they never actually leave the state. Once history exceeds a threshold, the `summary` node folds older turns into a running summary (merged with the prior summary, not overwritten) and removes them from `chat\_history`, keeping only the most recent few turns verbatim. This keeps token usage roughly constant instead of growing linearly with conversation length.



\## Tech stack



\- \*\*LangGraph\*\* — state graph / agent orchestration

\- \*\*LangChain + Gemini 2.5 Flash\*\* — LLM calls, structured output for relevance grading

\- \*\*Chroma\*\* — local vector store

\- \*\*BAAI/bge-small-en-v1.5\*\* (HuggingFace) — embedding model

\- \*\*Streamlit\*\* — chat UI + PDF upload/ingestion



\## Setup



```bash

git clone https://github.com/<your-username>/rag-conversational-chatbot.git

cd rag-conversational-chatbot

pip install -r requirements.txt

cp .env.example .env   # then add your GEMINI\_API\_KEY

streamlit run app1.py

```



Upload a PDF in the sidebar, click \*\*Ingest\*\*, then chat.



\## Known limitations / next steps



\- Guardrail is a keyword blocklist — good enough for a demo, not robust against paraphrased injection attempts. A classifier-based guardrail would be a natural upgrade.

\- No persistence layer (`MemorySaver`/checkpointer) yet — conversation state lives only in the Streamlit session.

\- No automated evaluation of retrieval/answer quality yet (would add RAGAS or a small hand-labeled QA set next).

\- Single shared Chroma collection — no per-user isolation, fine for a demo, not for multi-tenant use.



\## Project structure



├── app1.py # Streamlit UI

├── agent.py # LangGraph graph definition

├── nodes.py # Node implementations (guardrail, retrieval, generation, memory, etc.)

├── graph\_state.py # Shared state schema (TypedDict)

├── ingest.py # PDF ingestion → chunking → embedding → Chroma

├── embeddings.py # Shared embedding model instance

└── requirements.txt

