# Conversational-AI-with-LangGraph-RAG
Stateful AI assistant using LangGraph, LangChain,RAG, FAISS, SQLite memory, and Groq LLMs with ChatGPT-style streaming conversations.
conversational AI system inspired by ChatGPT architecture, built using LangGraph, LangChain, Streamlit, and FAISS.

This project replicates several core capabilities of modern AI assistants like ChatGPT:

* 💬 Stateful conversations
* 🧠 Persistent memory
* 📄 Retrieval-Augmented Generation (RAG)
* 🛠️ Tool calling and reasoning
* ⚡ Real-time response streaming
* 📚 Document understanding
* 🧵 Multi-thread conversations
* 🔄 Workflow-based AI orchestration

---

# 🚀 Project Vision

Modern conversational AI systems are not just single LLM calls.

They combine:

* Memory systems
* Retrieval pipelines
* Tool execution
* Stateful workflows
* Streaming architectures
* Context management

This project is an attempt to engineer those concepts from scratch using open-source GenAI frameworks.

The goal is to deeply understand how systems like ChatGPT can be architected and orchestrated in real-world applications.

---

# ✨ Core Features

# 💬 ChatGPT-Style Conversational Interface

### Features

* Persistent conversation memory
* Context-aware responses
* Real-time token streaming
* Conversation restoration
* Multi-session chat handling

### Powered By

* LangGraph state management
* SQLite checkpoint persistence
* Streamlit interactive UI

---

# 📄 Retrieval-Augmented Generation (RAG)

Upload any PDF and interact with it conversationally.

The assistant:

* Parses documents
* Splits text intelligently
* Generates embeddings
* Stores vectors in FAISS
* Retrieves semantically relevant chunks
* Injects context into the LLM

### Supported Use Cases

* Research papers
* Notes
* Documentation
* Technical PDFs
* Study material
* Reports

---

# 🛠️ AI Tool Calling System

The assistant dynamically decides when to use tools.

## Available Tools

| Tool              | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `calculator`      | Mathematical calculations                |
| `get_stock_price` | Live stock market information            |
| `Rag_tool`        | Retrieve context from uploaded documents |

---

# 🧠 LangGraph Workflow Architecture

This project uses graph-based AI orchestration instead of simple sequential pipelines.

## Workflow Graph

```text
                ┌────────────┐
                │   START    │
                └─────┬──────┘
                      ↓
             ┌────────────────┐
             │   chat_node    │
             └──────┬─────────┘
                    ↓
         ┌──────────────────────┐
         │   tools_condition    │
         └─────────┬────────────┘
                   ↓
            ┌─────────────┐
            │  ToolNode   │
            └──────┬──────┘
                   ↓
             ┌────────────┐
             │ chat_node  │
             └─────┬──────┘
                   ↓
               ┌──────┐
               │ END  │
               └──────┘
```

---

# 🧠 Why LangGraph?

Traditional chatbots:

* Stateless
* Linear execution
* Limited orchestration

LangGraph enables:

* Stateful workflows
* Conditional execution
* Tool routing
* Persistent memory
* Multi-step reasoning
* Agentic architectures

This is much closer to how modern production AI systems operate.

---

# 🏗️ Tech Stack

| Technology                     | Role                         |
| ------------------------------ | ---------------------------- |
| Python                         | Core programming language    |
| LangGraph                      | AI workflow orchestration    |
| LangChain                      | LLM tooling and abstractions |
| Streamlit                      | Frontend chat interface      |
| FAISS                          | Vector similarity search     |
| SQLite                         | Persistent checkpoint memory |
| HuggingFace Embeddings         | Semantic embeddings          |
| Groq API                       | High-speed LLM inference     |
| PyPDFLoader                    | PDF ingestion                |
| RecursiveCharacterTextSplitter | Document chunking            |

---

# 📂 Project Structure

```bash
project/
│
├── chatbot_with_rag.py
│      # Core LangGraph workflow
│      # Tool definitions
│      # RAG pipeline
│      # Memory management
│
├── app.py
│      # Streamlit frontend
│      # Chat interface
│      # Streaming UI
│      # Thread handling
│
├── chatbot.db
│      # SQLite persistent memory
│
├── requirements.txt
│
├── .env
│
└── README.md
```

---

# ⚙️ Installation Guide

# 1️⃣ Clone Repository

```bash
git clone https://github.com/Prathameshshelke145/Conversational-AI-with-LangGraph-RAG.git
cd Conversational-AI-with-LangGraph-RAG
```

---

# 2️⃣ Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 🧠 Internal System Architecture

# 🔹 Conversational Memory System

Conversation memory is persisted using SQLite checkpointing.

```python
conn=sqlite3.connect("chatbot.db",check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
```

This enables:

* Long-running conversations
* Persistent chat history
* Conversation restoration
* Stateful AI interactions

---

# 🔹 RAG Pipeline Architecture

```text
                 PDF Upload
                      ↓
               PyPDFLoader
                      ↓
             Text Chunking
                      ↓
       HuggingFace Embeddings
                      ↓
              FAISS Vector DB
                      ↓
                 Retriever
                      ↓
            Context Injection
                      ↓
                    LLM
```

---

# 🔹 Streaming Response System

The assistant streams responses token-by-token similar to ChatGPT.

```python
chatbot.stream(
    stream_mode="messages"
)
```

Benefits:

* Faster perceived response time
* Better user experience
* Real-time AI interaction

---

# 💬 Multi-Thread Conversation System

Each conversation is assigned a unique UUID-based thread ID.

## Advantages

* Independent chat sessions
* Persistent conversations
* Conversation switching
* Long-term memory handling

---

# 📄 Smart Document Understanding

The assistant automatically detects:

* PDF-related questions
* Document context requests
* File-based queries

It then dynamically invokes the RAG retrieval tool.

This simulates how production AI systems combine retrieval with reasoning.

---

# 🧪 Example Queries

# 💬 General Conversation

```text
Explain transformers in deep learning
```

```text
How does attention mechanism work?
```

---

# 📄 PDF-Based Questions

```text
Summarize chapter 3
```

```text
What does the uploaded document say about neural networks?
```

---

# 🛠️ Tool Calling

```text
Multiply 45 by 8
```

```text
Get stock price of Tesla
```

---

# 🔥 What Makes This Project Interesting?

This is not a basic chatbot.

It demonstrates:

* AI workflow orchestration
* Stateful graph execution
* Tool-augmented reasoning
* Retrieval systems
* Persistent memory
* Streaming architecture
* Multi-thread conversation handling
* Production-style GenAI engineering

---

# 📈 Future Improvements

## Planned Features

* 🌐 Web search integration
* 🤖 Multi-agent systems
* 🧠 Long-term vector memory
* 🎙️ Voice assistant support
* 🧩 MCP integration
* 🔐 Authentication system
* ☁️ Cloud deployment
* 📚 Citation-aware RAG
* 🔍 Hybrid retrieval (BM25 + Vector)
* 👨‍💻 Human-in-the-loop workflows
* 🧠 Autonomous planning agents

---

# 🎯 Learning Outcomes

This project helped explore:

* LangGraph architecture
* Agentic workflows
* Tool calling
* RAG systems
* Memory persistence
* Vector databases
* Streaming AI systems
* Stateful orchestration
* Conversational AI engineering

---

# 🤝 Contributing

Pull requests and improvements are welcome.

## Contribution Steps

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

# 📜 License

MIT License

---

# 👨‍💻 Author

## Prathamesh Shelke

Built using:

* [LangGraph](https://www.langchain.com/langgraph?utm_source=chatgpt.com)
* [LangChain](https://www.langchain.com?utm_source=chatgpt.com)
* [Streamlit](https://streamlit.io?utm_source=chatgpt.com)
* [FAISS](https://faiss.ai?utm_source=chatgpt.com)
* [Groq](https://groq.com?utm_source=chatgpt.com)
