"""
agent.py
--------
Core Agentic AI logic: wires together
  - an open-source LLM served locally via Ollama (e.g. llama3.1, qwen2.5, mistral)
  - the Tool Calling layer (tools.py)
  - short-term + long-term Memory (memory.py)

This matches the "Agent + Tools + Memory" capability listed for the
AI E-Commerce Customer Support Agent use case.
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import ALL_TOOLS
from memory import ConversationMemory, LongTermMemory

SYSTEM_PROMPT = """You are "ShopAssist", an AI customer support agent for an
e-commerce store. You help customers with:
  - product questions (price, stock, description)
  - order status and tracking
  - return eligibility and initiating returns
  - escalating complex or sensitive issues to a human

Guidelines:
- Always use the available tools to fetch real data — never guess order
  status, stock, or prices.
- Before initiating a return, ALWAYS call check_return_eligibility first.
- If the customer is angry, reports fraud, or asks for a human, use
  escalate_to_human.
- Be concise, polite, and confirm details (order ID, product) back to the
  customer before taking any state-changing action like initiating a return.
- If you don't have enough information (e.g. missing order ID), ask for it.
"""


class EcommerceSupportAgent:
    def __init__(self, customer_email: str, model_name: str = "llama3.1"):
        self.customer_email = customer_email

        # Open-source LLM served locally by Ollama.
        # Install Ollama (https://ollama.com) and run: `ollama pull llama3.1`
        self.llm = ChatOllama(model=model_name, temperature=0.2)

        self.conversation_memory = ConversationMemory()
        self.long_term_memory = LongTermMemory(customer_id=customer_email)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("system", "Known context about this customer: {customer_context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, ALL_TOOLS, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=False,
            handle_parsing_errors=True,
        )

    def chat(self, user_message: str) -> str:
        # Pull relevant long-term memories for this query (semantic recall)
        recalled_facts = self.long_term_memory.recall(user_message)
        customer_context = "; ".join(recalled_facts) if recalled_facts else "No prior notes."

        result = self.executor.invoke({
            "input": user_message,
            "chat_history": self.conversation_memory.get_history(),
            "customer_context": customer_context,
        })

        response_text = result["output"]

        # Update short-term memory
        self.conversation_memory.add_user_message(user_message)
        self.conversation_memory.add_ai_message(response_text)

        # Heuristic: persist noteworthy facts to long-term memory
        self._maybe_remember(user_message, response_text)

        return response_text

    def _maybe_remember(self, user_message: str, response_text: str):
        """Very simple heuristic to decide what's worth remembering long-term.
        In production, you might use an LLM call to extract durable facts."""
        triggers = ["return initiated", "escalated", "prefer", "damaged", "complaint"]
        combined = (user_message + " " + response_text).lower()
        if any(t in combined for t in triggers):
            self.long_term_memory.remember(
                f"On this interaction, note: {user_message.strip()}"
            )
