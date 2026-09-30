from typing import List, Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class BaseAgent:
    """Base AI Agent wrapper providing message history handling and tool integration."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        tools: Optional[List[Any]] = None,
        system_prompt: str = "You are a helpful enterprise AI assistant.",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.chat_history: List[BaseMessage] = []

        # Initialize base model
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)

        # Bind tools if available
        if self.tools:
            self.llm_bound = self.llm.bind_tools(self.tools)
        else:
            self.llm_bound = self.llm

    def execute(self, user_input: str) -> str:
        """Executes a single conversational turn with message context."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | self.llm_bound
        response = chain.invoke({"history": self.chat_history, "input": user_input})

        # Persist conversation state
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response.content))

        return str(response.content)

    def reset_history(self) -> None:
        """Clears stored session history."""
        self.chat_history.clear()
