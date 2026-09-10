"""
memory.py
---------
Memory layer for the agent, combining two open-source memory types:

1. SHORT-TERM (conversation) memory — the running chat history for the
   current session, so the agent remembers what was just discussed
   ("that order", "the same product", etc.)

2. LONG-TERM (semantic) memory — a local ChromaDB vector store that
   persists key facts/preferences across sessions per customer
   (e.g. "customer prefers email updates", "customer had a damaged item
   last month"), retrieved via open-source sentence-transformer embeddings.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.messages import HumanMessage, AIMessage

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_memory")


class ConversationMemory:
    """Simple in-session rolling message history."""

    def __init__(self, max_turns: int = 12):
        self.messages = []
        self.max_turns = max_turns

    def add_user_message(self, text: str):
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai_message(self, text: str):
        self.messages.append(AIMessage(content=text))
        self._trim()

    def _trim(self):
        # keep only the most recent N turns to control context size
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]

    def get_history(self):
        return self.messages


class LongTermMemory:
    """
    Persistent semantic memory using ChromaDB (open source, runs locally,
    no external service needed) with the open-source MiniLM embedding model.
    """

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="customer_memory",
            embedding_function=self.embedder,
        )

    def remember(self, fact: str):
        """Store a durable fact about this customer (preference, past issue, etc.)."""
        doc_id = f"{self.customer_id}-{abs(hash(fact))}"
        self.collection.upsert(
            documents=[fact],
            metadatas=[{"customer_id": self.customer_id}],
            ids=[doc_id],
        )

    def recall(self, query: str, k: int = 3) -> list:
        """Retrieve the most relevant remembered facts about this customer."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where={"customer_id": self.customer_id},
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception:
            return []
