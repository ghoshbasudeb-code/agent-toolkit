
# agent-toolkit

# Agent Toolkit (`agent-toolkit`)

A lightweight Python SDK for building, extending, and deploying LLM-powered AI agents using LangChain and OpenAI.

---

## 📋 Features

- **BaseAgent Wrapper:** Simple conversational agent loop managing message history, system prompts, and tool binding.
- **Built-in Tools:**
  - **Data Analysis:** `calculate_metrics`, `format_json_response`
  - **Web Search:** `search_web` (via Tavily)
  - **API Integration:** `fetch_api_data` (HTTP GET/POST)
  - **Database Access:** `execute_sql_query` (Read-only SQL queries)
  - **Knowledge Base Retrieval (RAG):** `query_knowledge_base` (FAISS vector store)
- **Vectorstore Generation:** Script included to build vector indexes from custom documents.

---

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/ghoshbasudeb-code/agent-toolkit.git](https://github.com/ghoshbasudeb-code/agent-toolkit.git)
   cd agent-toolkit
