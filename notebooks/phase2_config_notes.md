# Phase 2 — Config Parameters Notes

> Companion to `chatbot2_config.ipynb` · No tools, no chat history — just config

---

## Why Config Matters

The same model with different config produces very different results. Before reaching for prompt engineering tricks, understanding config gives you direct, predictable control over:

- How creative or deterministic the output is
- How long the response can be
- When generation stops
- Which tokens are even considered at each step

---

## The 4 Core Parameters

---

### 1. `temperature`

**What it does:** Scales the probability distribution before sampling. Lower = sharper (one token dominates). Higher = flatter (more tokens are plausible).

**Range:**

| Provider | Range | Default |
|---|---|---|
| OpenAI | 0.0 – 2.0 | 1.0 |
| Anthropic | 0.0 – 1.0 | 1.0 |
| Gemini | 0.0 – 2.0 | varies by model |

**Practical guide:**

| Value | Use case |
|---|---|
| `0.0` | Deterministic — classification, yes/no, structured extraction |
| `0.2–0.4` | Focused — Q&A, summarization, technical explanations |
| `0.7` | Balanced — general assistant, good default |
| `1.0` | Creative — brainstorming, varied outputs |
| `1.5+` | Experimental — may produce incoherent output |

**Important:** At `temperature=0.0`, re-running the same prompt will produce the same output every time (for a given model version). This is useful for reproducibility in testing.

**Code pattern:**
```python
# OpenAI
response = client.responses.create(..., temperature=0.3)

# Anthropic
response = client.messages.create(..., temperature=0.3)

# Gemini
CONFIG = types.GenerateContentConfig(..., temperature=0.3)
```

---

### 2. `max_tokens` / `max_output_tokens`

**What it does:** Hard cap on the number of tokens the model can generate. Generation stops immediately when this limit is hit — even mid-sentence.

**Parameter name by provider:**

| Provider | Parameter name | Required? |
|---|---|---|
| OpenAI | `max_output_tokens` | No (has default) |
| Anthropic | `max_tokens` | **Yes — always required** |
| Gemini | `max_output_tokens` | No (has default) |

> ⚠️ Anthropic is the only provider that will **error** if you omit `max_tokens`. The other two fall back to their own defaults. Always include it to be explicit and avoid surprises.

**Sizing guide:**

| Task | Suggested tokens |
|---|---|
| Single word / classification | 10–20 |
| Short answer | 100–200 |
| Explanation / summary | 300–600 |
| Long-form content | 1000–2000 |
| Full code generation | 1500–4000 |

**How to know if the response was cut off:** Check the stop reason (see below). If it says `max_tokens` (Anthropic) or `MAX_TOKENS` (Gemini), the model had more to say but hit the limit.

**Code pattern:**
```python
# OpenAI
response = client.responses.create(..., max_output_tokens=300)

# Anthropic
response = client.messages.create(..., max_tokens=300)  # required

# Gemini
CONFIG = types.GenerateContentConfig(..., max_output_tokens=300)
```

---

### 3. `top_p` — Nucleus Sampling

**What it does:** Before sampling the next token, sorts all tokens by probability and keeps only the smallest set that sums to `top_p`. Tokens outside that set are excluded entirely.

**Range:** `0.0` – `1.0` across all three providers.

| Value | Effect |
|---|---|
| `1.0` | No cutoff — all tokens considered |
| `0.9` | Exclude the bottom 10% of probability mass |
| `0.5` | Only consider the top half of probability mass |
| `0.1` | Very narrow — only the most likely tokens |

**`temperature` vs `top_p`:**

Both control randomness, but from different angles:
- `temperature` reshapes the distribution (makes it sharper or flatter)
- `top_p` truncates the distribution (cuts off the tail)

They interact in complex ways. **Best practice: fix one and vary the other.** Most practitioners use `temperature` and leave `top_p=1.0`, or use `top_p` and leave `temperature=1.0`.

**Code pattern:**
```python
# OpenAI
response = client.responses.create(..., top_p=0.9, temperature=1.0)

# Anthropic
response = client.messages.create(..., top_p=0.9)
# Anthropic docs say: use top_p OR temperature, not both

# Gemini
CONFIG = types.GenerateContentConfig(..., top_p=0.9, temperature=1.0)
```

---

### 4. Stop Sequences

**What it does:** A list of strings that immediately halt generation when encountered. The stop string is **not** included in the output.

**Parameter name by provider:**

| Provider | Parameter name | Max sequences |
|---|---|---|
| OpenAI | `stop` | 4 |
| Anthropic | `stop_sequences` | up to 8192 chars total |
| Gemini | `stop_sequences` | 5 |

**When to use stop sequences:**

- **Structured output** — stop after a closing tag: `stop=["</answer>"]`
- **Numbered lists** — stop before item N: `stop=["4."]`
- **Dialogue formatting** — stop when the other speaker's turn starts: `stop=["User:"]`
- **Preventing overgeneration** — stop before unnecessary caveats: `stop=["Note:", "Disclaimer:"]`
- **JSON extraction** — stop after the closing brace: `stop=["}"]`

**Example — extracting just a JSON object:**
```python
SYSTEM_PROMPT = "Return only a JSON object, nothing else."
USER_QUESTION = "Extract: name=Alice, age=30"

# With stop at "}" + 1 char, we capture exactly the JSON
CONFIG = {"stop": ["\n\n"], "temperature": 0.0}
```

**Code pattern:**
```python
# OpenAI
response = client.responses.create(..., stop=["4.", "END"])

# Anthropic
response = client.messages.create(..., stop_sequences=["4.", "END"])

# Gemini
CONFIG = types.GenerateContentConfig(..., stop_sequences=["4.", "END"])
```

---

## Gemini-Only: `top_k`

Gemini exposes an additional sampling parameter that OpenAI and Anthropic do not:

**`top_k`** — at each generation step, only consider the top K most probable tokens. The model samples from those K tokens only.

| Value | Effect |
|---|---|
| `1` | Greedy — always picks the single most likely token (same as temperature=0) |
| `10` | Narrow — only 10 candidates at each step |
| `40` | Moderate (Gemini's default) |
| `100+` | Wide — closer to unrestricted sampling |

**`top_k` vs `top_p`:**
- `top_k` is a fixed count — always exactly K candidates
- `top_p` is adaptive — the count varies based on the probability distribution

Gemini applies them in sequence: `top_k` first (hard count limit), then `top_p` on the remaining set.

```python
# Gemini only
CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    top_k=40,
    top_p=0.9
)
```

---

## Checking Why Generation Stopped

Always worth inspecting after a call — especially when debugging cut-off responses.

```python
# OpenAI
print(response.status)                          # "completed" or "incomplete"

# Anthropic — most informative
print(response.stop_reason)
# "end_turn"      → finished naturally
# "max_tokens"    → hit the limit (was cut off)
# "stop_sequence" → a stop string was matched

# Gemini
print(response.candidates[0].finish_reason)
# "STOP"        → finished naturally
# "MAX_TOKENS"  → hit the limit
# "OTHER"       → unusual stop
```

Anthropic's `stop_reason` is the most explicit and useful for debugging. Make it a habit to log it during development.

---

## Token Usage — Tracking Cost

All three providers return token counts. Useful for cost estimation and debugging.

```python
# OpenAI
print(response.usage.input_tokens)
print(response.usage.output_tokens)

# Anthropic
print(response.usage.input_tokens)
print(response.usage.output_tokens)

# Gemini
print(response.usage_metadata.prompt_token_count)
print(response.usage_metadata.candidates_token_count)
```

**Rough cost intuition (check current pricing — rates change):**
- Input tokens are always cheaper than output tokens
- `max_tokens` caps your *maximum* output cost per call
- At `temperature=0`, output length is deterministic — useful for cost forecasting

---

## Full Config Reference — All 3 Providers

```python
# ── OpenAI ─────────────────────────────────────────────
CONFIG = {
    "temperature": 0.7,          # 0–2, default 1.0
    "max_output_tokens": 500,    # no default — be explicit
    "top_p": 1.0,                # 0–1, default 1.0
    "stop": ["END", "\n\n"]      # list, max 4 strings
}
response = client.responses.create(
    model="gpt-4o-mini",
    instructions=SYSTEM_PROMPT,
    input=USER_QUESTION,
    **CONFIG
)
text = response.output_text

# ── Anthropic ──────────────────────────────────────────
CONFIG = {
    "max_tokens": 500,           # REQUIRED — no default
    "temperature": 0.7,          # 0–1
    "top_p": 1.0,                # 0–1
    "stop_sequences": ["END"]    # list
}
response = client.messages.create(
    model="claude-sonnet-4-5",
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": USER_QUESTION}],
    **CONFIG
)
text = response.content[0].text

# ── Gemini ─────────────────────────────────────────────
CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.7,             # 0–2
    max_output_tokens=500,
    top_p=1.0,                   # 0–1
    top_k=40,                    # Gemini only
    stop_sequences=["END"]
)
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=USER_QUESTION,
    config=CONFIG
)
text = response.text
```

---

## What's Next

Now that you understand the core config layer, the natural next steps from your roadmap:

- **Phase 3 — Prompting:** System prompt vs user prompt, few-shot, structured output
- **Phase 4 — Native features:** Add chat history to these same API calls, then add tools

The config knowledge transfers directly — when you add history and tools, the config parameters work exactly the same way.
