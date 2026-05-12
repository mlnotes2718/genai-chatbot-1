# LLM API Tutorial (OpenAI, Anthropic, Gemini)

This tutorial demonstrates the **common structure** for using modern LLM APIs.

We will progressively build:

1. Basic API call
2. Add chat history
3. Add native web search

---

# 0. Setup Environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


# 1. Install Dependencies

```bash
uv add openai anthropic google-genai python-dotenv
```

---

# 2. Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_google_key
```

---

# 3. Common API Pattern

Use this mental model for every provider:

```text
client
   handles connection

model
   chooses capability

system prompt
   defines behavior

tools
   gives abilities

config
   tunes generation

user question
   defines task
```

Recommended structure:

```python
# 1. Client
client = ProviderClient()

# 2. Model
MODEL = "..."

# 3. Default prompt
SYSTEM_PROMPT = "You are a helpful assistant."

# 4. Optional tools
TOOLS = [...]

# 5. Optional generation settings
CONFIG = {...}

# 6. Dynamic user question
USER_QUESTION = "..."

# 7. API request
response = ...
```

---

# Part 1 — OpenAI

Official docs: https://platform.openai.com/docs

---

## 1.1 Basic API Call

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. Client
# -------------------
client = OpenAI()

# -------------------
# 2. Model
# -------------------
MODEL = "gpt-4o-mini"

# -------------------
# 3. System Prompt
# -------------------
SYSTEM_PROMPT = "You are a helpful assistant."

# -------------------
# 4. Tools
# -------------------
TOOLS = []

# -------------------
# 5. Config
# -------------------
CONFIG = {
    "temperature": 0.7
}

# -------------------
# 6. User Question
# -------------------
USER_QUESTION = "Explain what an LLM is."

# -------------------
# 7. Request
# -------------------
response = client.responses.create(
    model=MODEL,
    instructions=SYSTEM_PROMPT,
    tools=TOOLS,
    input=USER_QUESTION,
    **CONFIG
)

print(response.output_text)
```

---

## 1.2 Add Chat History

OpenAI is **stateless**, so we store history manually.

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

MODEL = "gpt-5"
SYSTEM_PROMPT = "You are a helpful assistant."

history = []

while True:
    USER_QUESTION = input("You: ")

    history.append({
        "role": "user",
        "content": USER_QUESTION
    })

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=history
    )

    answer = response.output_text
    print("AI:", answer)

    history.append({
        "role": "assistant",
        "content": answer
    })
```

---

## 1.3 Add Native Web Search

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

MODEL = "gpt-5"

SYSTEM_PROMPT = "You are a helpful assistant."

TOOLS = [
    {
        "type": "web_search"
    }
]

USER_QUESTION = "What are the latest AI news today?"

response = client.responses.create(
    model=MODEL,
    instructions=SYSTEM_PROMPT,
    tools=TOOLS,
    input=USER_QUESTION
)

print(response.output_text)
```

---

# Part 2 — Anthropic Claude

Official docs: https://docs.anthropic.com

---

## 2.1 Basic API Call

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. Client
# -------------------
client = anthropic.Anthropic()

# -------------------
# 2. Model
# -------------------
MODEL = "claude-sonnet-4"

# -------------------
# 3. System Prompt
# -------------------
SYSTEM_PROMPT = "You are a helpful assistant."

# -------------------
# 4. Tools
# -------------------
TOOLS = []

# -------------------
# 5. Config
# -------------------
CONFIG = {
    "max_tokens": 500,
    "temperature": 0.7
}

# -------------------
# 6. User Question
# -------------------
USER_QUESTION = "Explain transformers simply."

# -------------------
# 7. Request
# -------------------
response = client.messages.create(
    model=MODEL,
    system=SYSTEM_PROMPT,
    messages=[
        {
            "role": "user",
            "content": USER_QUESTION
        }
    ],
    **CONFIG
)

print(response.content[0].text)
```

---

## 2.2 Add Chat History

Anthropic is also **stateless**.

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4"
SYSTEM_PROMPT = "You are a helpful assistant."

history = []

while True:
    USER_QUESTION = input("You: ")

    history.append({
        "role": "user",
        "content": USER_QUESTION
    })

    response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=500,
        messages=history
    )

    answer = response.content[0].text
    print("Claude:", answer)

    history.append({
        "role": "assistant",
        "content": answer
    })
```

---

## 2.3 Add Native Web Search

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4"

SYSTEM_PROMPT = "You are a helpful assistant."

TOOLS = [
    {
        "name": "web_search"
    }
]

USER_QUESTION = "Find recent developments in multimodal AI."

response = client.messages.create(
    model=MODEL,
    system=SYSTEM_PROMPT,
    tools=TOOLS,
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": USER_QUESTION
        }
    ]
)

print(response.content[0].text)
```

---

# Part 3 — Google Gemini

Official docs: https://ai.google.dev/gemini-api/docs

---

## 3.1 Basic API Call

```python
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. Client
# -------------------
client = genai.Client()

# -------------------
# 2. Model
# -------------------
MODEL = "gemini-2.5-flash"

# -------------------
# 3. System Prompt
# -------------------
SYSTEM_PROMPT = "You are a helpful assistant."

# -------------------
# 4. Config
# -------------------
CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.7
)

# -------------------
# 5. User Question
# -------------------
USER_QUESTION = "Explain embeddings simply."

# -------------------
# 6. Request
# -------------------
response = client.models.generate_content(
    model=MODEL,
    contents=USER_QUESTION,
    config=CONFIG
)

print(response.text)
```

---

## 3.2 Add Chat History

Gemini supports **native chat sessions**.

```python
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = "You are a helpful assistant."

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT
    )
)

while True:
    USER_QUESTION = input("You: ")

    response = chat.send_message(USER_QUESTION)

    print("Gemini:", response.text)
```

---

## 3.3 Add Native Google Search

```python
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = "You are a helpful assistant."

SEARCH_TOOL = types.Tool(
    google_search=types.GoogleSearch()
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[SEARCH_TOOL]
)

USER_QUESTION = "What happened in AI this week?"

response = client.models.generate_content(
    model=MODEL,
    contents=USER_QUESTION,
    config=CONFIG
)

print(response.text)
```

---

# Side-by-Side Comparison

| Concept | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| System Prompt | `instructions=` | `system=` | `system_instruction=` |
| Model | `model=` | `model=` | `model=` |
| Tools | `tools=` | `tools=` | `tools=` |
| Config | `dict` | `dict` | `GenerateContentConfig()` |
| User Input | `input=` | `messages=` | `contents=` |
| Chat History | Manual | Manual | Native Chat |
| Web Search | `web_search` | `web_search` | `GoogleSearch()` |

---

# Key Takeaways

Always structure your LLM code like this:

```python
client
MODEL
SYSTEM_PROMPT
TOOLS
CONFIG
USER_QUESTION
response
```

This keeps your code:

- readable
- reusable
- provider-independent
- easy to upgrade

The API syntax changes, but the architecture stays the same.
