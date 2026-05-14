# Phase 2 — LLM Vendor API Notes

> Companion notes for `chatbot1.ipynb` · Based on your notebook code + vendor docs

---

## Your Notebook at a Glance

Your notebook is well-structured. The consistent 7-step pattern across all three providers is exactly the right mental model — it makes switching vendors mechanical rather than confusing.

**✅ What's solid in your notebook:**
- Consistent `client → MODEL → SYSTEM_PROMPT → TOOLS → CONFIG → USER_QUESTION → response` structure across all three providers
- Correct use of `dotenv` for API key management
- Chat history loop (OpenAI) is implemented correctly — appending both user and assistant turns
- GPT-4 vs GPT-5 differences are well documented with the `reasoning.effort` and `text.verbosity` shift
- Side-by-side comparison table at the end is a great reference

**📝 One thing to note:**
The Anthropic and Gemini sections show basic API calls but don't include the full chat history loop yet (unlike OpenAI). That's fine for learning — just know that pattern is the same for all three: append user message → call API → append assistant response.

---

## The Common Pattern — Why It Matters

Your notebook shows this abstraction for every provider:

```python
client        # handles auth + connection
MODEL         # selects capability tier
SYSTEM_PROMPT # defines persistent behavior
TOOLS         # grants external abilities
CONFIG        # tunes generation parameters
USER_QUESTION # the dynamic task
response      # parse the output
```

This is the right way to think about it. **The API syntax changes; the architecture doesn't.** Once you internalize this, switching providers is just syntax substitution.

---

## Provider Deep Dive

---

### OpenAI — `openai` SDK

**Docs:** https://platform.openai.com/docs
**Install:** `pip install openai`

#### API structure

OpenAI uses the **Responses API** (their newer interface, which your notebook correctly uses):

```python
response = client.responses.create(
    model=MODEL,
    instructions=SYSTEM_PROMPT,   # ← OpenAI calls it "instructions"
    tools=TOOLS,
    input=history,                # ← accepts string OR list of messages
    **CONFIG
)

print(response.output_text)       # ← simple text access
```

#### Config parameters

**GPT-4o / GPT-4o-mini** (randomness-controlled):
```python
CONFIG = {
    "temperature": 0.7,       # 0 = deterministic, >1 = creative
    "max_output_tokens": 500
}
```

**GPT-5 / GPT-5-nano** (reasoning-controlled — your notebook covers this well):
```python
CONFIG = {
    "reasoning": {
        "effort": "low"       # minimal | low | medium | high
    },
    "text": {
        "verbosity": "medium" # low | medium | high
    },
    "max_output_tokens": 500
}
```

#### Chat history format

```python
history = [
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user",      "content": "Explain attention"}
]
```

Pass the full list to `input=history`. OpenAI's Responses API handles it automatically.

#### Web search

```python
TOOLS = [{"type": "web_search"}]
```

OpenAI's built-in web search. No extra config required — the model decides when to use it. Citations appear inline in `response.output_text`.

---

### Anthropic Claude — `anthropic` SDK

**Docs:** https://docs.anthropic.com
**Install:** `pip install anthropic`

#### API structure

```python
response = client.messages.create(
    model=MODEL,
    system=SYSTEM_PROMPT,         # ← Anthropic calls it "system"
    messages=[                    # ← always a list, even for one turn
        {"role": "user", "content": USER_QUESTION}
    ],
    **CONFIG
)

print(response.content[0].text)   # ← response is block-based, not a single string
```

**Key difference from OpenAI:** The response is a list of content blocks (`response.content`), not a single string. This is because Claude can return mixed content — text blocks, tool use blocks, and tool result blocks in the same response. You must always index into `response.content[0].text` for plain text.

#### Config parameters

```python
CONFIG = {
    "max_tokens": 500,    # REQUIRED — Anthropic has no default, will error without it
    "temperature": 0.3
}
```

> ⚠️ **`max_tokens` is required** for every Anthropic call. OpenAI and Gemini have defaults; Anthropic does not. Forgetting this is the most common error when switching to Claude.

#### Chat history format

Same structure as OpenAI, but passed to `messages=` instead of `input=`:

```python
messages = [
    {"role": "user",      "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user",      "content": "Explain attention"}
]
```

The `system=` parameter stays separate from `messages=` — it's not part of the message array.

#### Web search

```python
TOOLS = [{"type": "web_search_20250305", "name": "web_search"}]
```

Anthropic's web search tool is versioned (note the date suffix `_20250305`). This is intentional — Anthropic versions tool APIs to allow safe evolution without breaking existing code. Always check docs for the latest version string.

#### Block-based response parsing

When tools are used, the response contains multiple block types:

```python
for block in response.content:
    if block.type == "text":
        print(block.text)
    elif block.type == "tool_use":
        print(f"Calling tool: {block.name}")
        print(f"With input: {block.input}")
```

This is unique to Anthropic — the other two providers return simpler structures for basic usage.

---

### Google Gemini — `google-genai` SDK

**Docs:** https://ai.google.dev/gemini-api/docs
**Install:** `pip install google-genai`

#### API structure

Gemini has a noticeably different style — config is a typed object, not a plain dict:

```python
from google.genai import types

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,  # ← lives inside the config object
    temperature=0.2
)

response = client.models.generate_content(
    model=MODEL,
    contents=USER_QUESTION,            # ← called "contents", not "input" or "messages"
    config=CONFIG
)

print(response.text)                   # ← simplest text access of the three
```

**Key difference:** The system prompt lives *inside* the config object (`system_instruction=`), not as a top-level parameter. This is the most common gotcha when porting Gemini code.

#### Config parameters

```python
CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.2,
    max_output_tokens=500,
    top_p=0.9,
    top_k=40                           # unique to Gemini — see below
)
```

#### Chat history — native chat session (unique to Gemini)

Your notebook notes this in the comparison table. Gemini has a **built-in stateful chat session**, unlike OpenAI and Anthropic which require you to manage history manually:

```python
chat = client.chats.create(model=MODEL, config=CONFIG)

response1 = chat.send_message("What is an LLM?")
response2 = chat.send_message("Can you give an example?")  # ← remembers context automatically
```

The `chat` object maintains the conversation history internally. You don't append messages manually. This is Gemini's native chat interface — you can still do manual history management with `generate_content` if you prefer consistency across providers.

#### Web search

```python
from google.genai import types

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(google_search=types.GoogleSearch())]  # ← strongly-typed tool object
)
```

Gemini's web search is deeply integrated — it's backed by Google Search directly, which gives it an edge for factual, real-time queries. The grounding metadata (sources, citations) is accessible via `response.candidates[0].grounding_metadata`.

---

## Side-by-Side API Comparison

| Concept | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| **SDK install** | `pip install openai` | `pip install anthropic` | `pip install google-genai` |
| **Client init** | `OpenAI()` | `anthropic.Anthropic()` | `genai.Client()` |
| **API method** | `client.responses.create()` | `client.messages.create()` | `client.models.generate_content()` |
| **System prompt param** | `instructions=` | `system=` | inside `GenerateContentConfig` |
| **User input param** | `input=` | `messages=` | `contents=` |
| **Config style** | plain `dict` | plain `dict` | typed `GenerateContentConfig()` |
| **max_tokens required?** | No (has default) | **Yes — required** | No (has default) |
| **Response text access** | `response.output_text` | `response.content[0].text` | `response.text` |
| **Chat history** | Manual (append list) | Manual (append list) | **Native chat session** |
| **Web search** | `{"type": "web_search"}` | `{"type": "web_search_20250305"}` | `GoogleSearch()` typed object |
| **Tool response format** | Simple | **Block-based** (text/tool_use/tool_result) | Simple |

---

## Unique Features — What Only One Provider Offers

### OpenAI only

**`reasoning.effort` control (GPT-5)**
The ability to explicitly set *how hard* the model thinks before responding — `minimal`, `low`, `medium`, `high` — with predictable latency/quality tradeoffs. No other vendor currently exposes this as a direct parameter.

**`text.verbosity` control (GPT-5)**
Separate from length limits, this controls the *style* of the output — whether the model is terse or expansive in its explanations. Anthropic and Gemini don't have an equivalent.

**`response.output_text` shortcut**
The simplest text extraction of the three — a single string, no indexing required. Small thing, but convenient.

---

### Anthropic only

**Block-based response content**
Every response is a list of typed content blocks. This is more verbose for simple use cases, but it's the only provider that gives you clean, structured access to *interleaved* tool calls and text in a single response. You can have: `[text_block, tool_use_block, tool_result_block, text_block]` in one response.

**Versioned tool APIs**
Tool type strings include a date suffix (e.g. `web_search_20250305`). This means Anthropic can evolve tool behavior without silently breaking your existing code — you opt in to new versions explicitly.

**`max_tokens` required by design**
While this feels like a constraint, it's intentional — Anthropic forces you to think about your budget upfront rather than getting surprise long (and expensive) responses.

**Extended thinking (Claude 3.7+)**
Claude can expose its internal reasoning chain as a separate `thinking` block before the final answer. No other major provider exposes this:

```python
CONFIG = {
    "max_tokens": 16000,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000   # how many tokens the model can "think" with
    }
}
# response.content will include a thinking block + a text block
```

---

### Gemini only

**Native stateful chat sessions**
`client.chats.create()` gives you a conversation object that manages history internally. This is the cleanest chat interface of the three — no manual list management. The other two providers require you to append messages yourself.

**`top_k` sampling parameter**
Gemini exposes `top_k` in its config (only consider the top-k most probable tokens). OpenAI and Anthropic don't expose this parameter directly.

**Google Search grounding (real-time)**
Because Gemini's web search is backed by Google Search directly, it has access to the same index and freshness that Google.com uses. OpenAI's and Anthropic's web search tools use their own crawlers/indexes. For factual, real-time queries, Gemini's grounding tends to be more comprehensive.

**Multimodal-first architecture**
While all three support images, Gemini was designed multimodal from the ground up — the same `contents=` field accepts text, images, audio, and video natively. Mixing modalities is more natural in Gemini's API.

**File API for large context**
Gemini supports uploading files (PDFs, audio, video) via a Files API, then referencing them by URI in your request. This is the cleanest way to handle very long documents or media files.

```python
# Upload once
file = client.files.upload(path="document.pdf")

# Reference in request
response = client.models.generate_content(
    model=MODEL,
    contents=[file, "Summarize this document"]
)
```

---

## When to Use Which Provider

| Situation | Best choice | Reason |
|---|---|---|
| General chatbot, customer support | Any — OpenAI GPT-4o-mini is cheapest | Commodity task |
| Complex multi-step reasoning / agents | OpenAI GPT-5 or Anthropic Claude | Reasoning control (GPT-5) or reliable tool use (Claude) |
| Long document analysis (100k+ tokens) | Anthropic Claude or Gemini | Large context windows, Gemini's File API |
| Real-time factual queries | Gemini | Google Search grounding |
| Interleaved tool calls with text | Anthropic | Block-based response handles this cleanly |
| Multimodal (audio, video, images) | Gemini | Multimodal-first design |
| Fastest prototyping | Gemini | Native chat session, `response.text` is simplest |
| Regulated / safety-critical apps | Anthropic | Constitutional AI training, most safety documentation |

---

## Quick Reference — Response Parsing

```python
# OpenAI
text = response.output_text

# Anthropic
text = response.content[0].text

# Gemini
text = response.text
```

```python
# Anthropic — full block parsing (for tool-use flows)
for block in response.content:
    if block.type == "text":
        text = block.text
    elif block.type == "tool_use":
        tool_name = block.name
        tool_input = block.input
```

---

## Environment Setup (from your notebook)

```env
# .env file
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_google_key
```

```python
from dotenv import load_dotenv
load_dotenv()

# Each SDK auto-reads its env var:
# OpenAI    → OPENAI_API_KEY
# Anthropic → ANTHROPIC_API_KEY
# Gemini    → GEMINI_API_KEY  (or pass api_key= explicitly)
```

---

## Recommended Next Steps

Your notebook covers Phase 2 well. When moving to Phase 3 (Prompting Fundamentals), the key things to experiment with directly in these APIs:

1. **System prompt vs user prompt** — try moving instructions between them and observe the difference in behavior
2. **Temperature** — run the same prompt at 0.0, 0.7, and 1.5 with all three providers
3. **Tool calling** — enable web search and ask something time-sensitive (today's news, live prices)
4. **Block parsing (Anthropic)** — ask Claude to use a tool and inspect the raw `response.content` list

---

*Phase 2 complete → Next: Phase 3 — Prompting Fundamentals*
