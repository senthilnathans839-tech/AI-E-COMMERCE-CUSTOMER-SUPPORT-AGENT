# ShopAssist — AI E-Commerce Customer Support Agent

Implementation of **Use Case #3: AI E-Commerce Customer Support Agent**
("Handles product queries, order-status requests, returns and recommendations" —
Key capabilities: **Tool Calling + Memory**) using a **fully open-source** stack.

## Architecture

```
User message
    │
    ▼
main.py  (CLI chat loop)
    │
    ▼
agent.py  (LangChain tool-calling agent)
    │        │
    │        ├── Short-term memory (ConversationMemory) → last N chat turns
    │        └── Long-term memory (ChromaDB + MiniLM embeddings) → durable
    │            customer facts (past returns, complaints, preferences)
    │
    ▼
Ollama LLM (llama3.1 / mistral / qwen2.5 — runs 100% locally)
    │
    ▼
tools.py  (Tool Calling layer)
    ├── search_product           → product catalog lookup
    ├── check_order_status       → order tracking
    ├── list_customer_orders     → find orders by email
    ├── check_return_eligibility → return-window check
    ├── initiate_return          → state-changing return action
    └── escalate_to_human        → hands off complex/sensitive cases
```

## Open-source components used

| Layer            | Technology                                   | Why                                   |
|-------------------|----------------------------------------------|----------------------------------------|
| LLM               | [Ollama](https://ollama.com) + Llama 3.1 / Mistral / Qwen2.5 | Free, runs locally, no API key |
| Agent framework   | [LangChain](https://python.langchain.com)     | Open-source tool-calling agent runtime |
| Vector memory     | [ChromaDB](https://www.trychroma.com)         | Open-source local vector database      |
| Embeddings        | `sentence-transformers` (all-MiniLM-L6-v2)    | Open-source embedding model            |
| Data store        | Local JSON (`data/products.json`, `data/orders.json`) | Stand-in for a real Order/Product API |

## Setup

1. **Install Ollama** and pull a model:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.1
   ollama serve   # starts the local LLM server
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the agent**:
   ```bash
   python main.py
   ```

## Example conversation

```
You: Is the Aroma Pro Wireless Earbuds in stock?
ShopAssist: Yes! The Aroma Pro Wireless Earbuds (₹2499) are in stock —
42 units available. They come with a 10-day return window.

You: What's the status of order ORD5001?
ShopAssist: Order ORD5001 (Aroma Pro Wireless Earbuds) is currently
Shipped, expected delivery on 2026-09-10. Tracking ID: TRK998877.

You: I want to return order ORD5002, it arrived damaged.
ShopAssist: Let me check — order ORD5002 was delivered on 2026-09-05 and
is within its 10-day return window, so it's eligible. I've initiated the
return with reason "arrived damaged". A pickup will be scheduled within
2-3 business days.

You: This is the second time I've had a damaged item, I want to talk to someone.
ShopAssist: I understand your frustration. I've escalated this to our
support team (Ticket ID: TCK48213). A human agent will contact you
within 24 hours.
```

## Swapping in a different open-source LLM

Change one line in `agent.py`:
```python
self.llm = ChatOllama(model="mistral", temperature=0.2)
# or "qwen2.5", "phi3", "gemma2", etc. — any model available via `ollama pull`
```

## Extending this for real production use

- Replace `data/products.json` / `data/orders.json` reads in `tools.py`
  with calls to your real Product Catalog API / Order Management System.
- Add a `RAG` tool over your FAQ/policy documents using ChromaDB (same
  pattern as `LongTermMemory`) for the "RAG" capability if you want the
  agent to also answer policy questions from documents.
- Deploy `agent.py` behind a FastAPI endpoint and connect it to your
  website chat widget or WhatsApp Business API.
- Swap ChromaDB for a hosted open-source vector DB (e.g. self-hosted
  Qdrant or Weaviate) if you need multi-server scaling.
