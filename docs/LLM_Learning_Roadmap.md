# LLM Study Roadmap

A practical roadmap for learning Large Language Models (LLMs), from basic concepts to building production-ready applications.

---

# Phase 1 — LLM Fundamentals

## Objective
Understand what an LLM is and how it works conceptually.

---

## Topics

### 1. Transformer Basics
Learn the architecture behind modern LLMs.

Topics:
- Tokens
- Embeddings
- Attention mechanism
- Self-attention
- Multi-head attention
- Positional encoding
- Encoder vs Decoder
- Autoregressive generation

Goal:
- Understand how text becomes predictions.

Recommended:
- "Attention Is All You Need"
- Illustrated Transformer (Jay Alammar)

---

### 2. Training Concepts

Topics:
- Pretraining
- Fine-tuning
- Instruction tuning
- RLHF
- Alignment
- Context window
- Token limits

Goal:
- Understand how LLMs are created.

---

### 3. Inference Concepts

Topics:
- Temperature
- Top-p
- Max tokens
- Deterministic vs stochastic output
- Sampling

Goal:
- Understand generation behavior.

---

# Phase 2 — Basic Vendor APIs

## Objective
Learn how to call LLMs directly using provider SDKs.

---

## Providers

### OpenAI API
Learn:
- client setup
- model selection
- system prompt
- user input
- config
- response parsing

Models:
- gpt-4o-mini
- gpt-5-nano

---

### Anthropic API
Learn:
- messages.create()
- system prompt
- tools
- block-based responses

Models:
- claude-sonnet

---

### Gemini API
Learn:
- generate_content()
- chat sessions
- generation config

Models:
- gemini-2.5-flash

---

## Common Pattern

```python
client
MODEL
SYSTEM_PROMPT
CONFIG
USER_QUESTION
response
```

Goal:
- Understand vendor-native APIs before abstractions.

---

# Phase 3 — Prompting Fundamentals

## Objective
Learn how to control model behavior.

---

## Topics

### 1. System Prompt vs User Prompt

```text
System prompt = permanent behavior
User prompt = current task
```

---

### 2. Zero-shot Prompting

```text
Explain transformers.
```

---

### 3. Role Prompting

```text
You are an expert ML tutor.
```

---

### 4. Output Formatting

```text
Explain in 3 bullet points.
```

---

### 5. Few-shot Prompting

```text
Input → Output examples
```

---

### 6. Structured Output Prompting

```text
Return JSON:
{
  "summary": "",
  "keywords": []
}
```

---

### 7. Prompt Iteration

Mindset:

```text
Prompting = debugging instructions
```

Goal:
- Learn to steer model behavior effectively.

---

# Phase 4 — Native LLM Features

## Objective
Learn common capabilities beyond simple prompting.

---

## 1. Chat History

Topics:
- message arrays
- user + assistant history
- manual memory
- conversation context

Example:

```python
history = [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

---

## 2. Web Search

Topics:
- built-in search tools
- tool definitions
- response parsing
- citations

Goal:
- Augment model with fresh information.

---

## 3. Structured Output

Topics:
- JSON mode
- schemas
- validation

Goal:
- Reliable machine-readable responses.

---

## 4. Function / Tool Calling

Topics:
- tool definitions
- arguments
- execution loop

Goal:
- Connect LLM to external systems.

---

# Phase 5 — Memory & Persistence

## Objective
Persist conversations beyond one session.

---

## Topics

### In-memory

```python
history = []
```

---

### Redis-backed chat memory

Learn:
- Redis connection
- session keys
- append messages
- expiration

Example:

```text
chat:user_001
```

Goal:
- Persistent conversational memory.

---

# Phase 6 — Multi-provider Abstractions

## Objective
Use one interface for many providers.

---

## 1. LiteLLM

Use when:
- switching vendors
- benchmarking
- fallback routing

Learn:

```python
completion(model="...")
```

Benefits:
- OpenAI-like interface
- many providers

---

## 2. Groq

Use when:
- low latency matters
- fast experimentation

Learn:
- Groq SDK
- hosted open-source models

---

## 3. LangChain LLM Wrappers

Learn:

```python
llm.invoke(...)
```

Benefits:
- standard interface
- integrates with LangChain ecosystem

---

# Phase 7 — RAG (Retrieval-Augmented Generation)

## Objective
Build systems that answer using external documents.

---

## Topics

### Document Loading
- PDFs
- Markdown
- Web pages

---

### Chunking
- recursive splitters
- overlap

---

### Embeddings
- embedding models
- vector representation

---

### Vector Databases
- Chroma
- FAISS
- Pinecone

---

### Retrieval
- similarity search
- top-k retrieval

---

### Prompt Injection of Context

```text
Context + Question
```

Goal:
- Answer grounded in your own data.

---

# Phase 8 — Chains & Agents

## Objective
Build multi-step reasoning workflows.

---

## LangChain Chains

Learn:
- prompt templates
- runnable chains
- pipe syntax

---

## Agents

Learn:
- planning
- tool selection
- iterative reasoning

---

## LangGraph

Learn:
- nodes
- edges
- state
- memory
- retry logic

Goal:
- production-grade agent systems.

---

# Phase 9 — Evaluation

## Objective
Measure LLM quality.

---

## Topics

- Human evaluation
- Prompt regression testing
- BLEU
- ROUGE
- Perplexity
- LLM-as-judge
- latency
- cost tracking

Goal:
- Improve reliability systematically.

---

# Phase 10 — Production Engineering

## Objective
Deploy robust LLM systems.

---

## Topics

### Security
- API key management
- prompt injection defense
- secret scanning

---

### Observability
- logging
- tracing
- token usage

Tools:
- LangSmith

---

### Cost Optimization
- model selection
- caching
- batching

---

### Reliability
- retries
- fallbacks
- timeout handling

---

### Deployment
- FastAPI
- Docker
- CI/CD

---

# Suggested Learning Order

```text
1. LLM Fundamentals
2. Basic Vendor APIs
3. Prompting Fundamentals
4. Native Features
5. Memory & Redis
6. LiteLLM / Groq / LangChain
7. RAG
8. Chains & Agents
9. Evaluation
10. Production
```

---

# Guiding Principle

```text
Learn raw APIs first.
Add abstractions later.
Understand prompting before tools.
Build simple before building agents.
```
