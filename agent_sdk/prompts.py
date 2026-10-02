from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Standard system instructions
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, precise enterprise AI assistant. "
    "Utilize available tools whenever necessary to answer questions accurately."
)

ANALYTICS_AGENT_PROMPT = (
    "You are an expert data analytics assistant. "
    "Analyze data trends, compute metrics, and format structured insights."
)

RAG_SEARCH_PROMPT = (
    "You are a knowledge-retrieval AI assistant. "
    "Base your responses strictly on the retrieved context from vector search. "
    "If information is missing, state clearly that you do not know."
)


def get_agent_prompt_template(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> ChatPromptTemplate:
    """Constructs a standard ChatPromptTemplate for LangChain agents."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
