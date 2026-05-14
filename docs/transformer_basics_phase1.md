# Phase 1 — Transformer Basics Study Material

> From the LLM Study Roadmap · Phase 1: LLM Fundamentals

---

## 1. Tokens

LLMs don't read words — they read **tokens**. A token is a chunk of text, roughly 3–4 characters on average. The model has a fixed vocabulary (e.g. ~100k tokens for GPT-4). Your text is split into tokens before anything else happens.

**Example:**
```
"unbelievable" → ["un", "believ", "able"]  — 3 tokens, not 1 word
```

| Text | Tokens | Token IDs |
|------|--------|-----------|
| The transformer architecture is unbelievable! | The / ▁transformer / ▁architecture / ▁is / ▁un / believ / able / ! | 464, 14121, 12382, 318, 555, 7357, 540, 0 |

**Key rule of thumb:** 1 token ≈ 4 characters ≈ 0.75 words in English.
A 4096-token context window holds roughly 3,000 words.

---

## 2. Embeddings

Each token ID is looked up in an **embedding table** to get a dense vector — typically 512 to 8192 numbers. Similar words land near each other in this high-dimensional space.

**Classic example:**
```
"king" - "man" + "woman" ≈ "queen"
```
The vector arithmetic captures meaning.

**How it works:**
- Token ID → lookup in embedding matrix → dense float vector
- Embedding matrix shape: `[vocab_size × d_model]`
- e.g. 100,000 tokens × 4096 dimensions = 400M parameters just for embeddings

**Key insight:** The model learns these values during training — no human assigns them. They emerge from predicting the next token billions of times.

---

## 3. Self-Attention

Attention lets every token look at every other token and decide how much to "attend" to it. The result is a weighted mix of information from the whole sequence — context flows everywhere.

**Example:**
In `"The bank by the river"`, attention helps `"bank"` know it means *riverbank*, not a *financial bank*, by attending strongly to `"river"`.

**The formula:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

- **Q (Query):** What am I looking for?
- **K (Key):** What do I contain?
- **V (Value):** What information do I pass along?

**Simplified attention weights for "The bank by the river":**

|        | The  | bank | by   | the  | river |
|--------|------|------|------|------|-------|
| The    | 0.10 | 0.15 | 0.05 | 0.10 | **0.60** |
| bank   | 0.05 | 0.20 | 0.10 | 0.05 | **0.60** |
| by     | 0.15 | 0.25 | 0.10 | 0.15 | 0.35 |
| the    | 0.10 | 0.10 | 0.05 | 0.15 | **0.60** |
| river  | 0.05 | 0.15 | 0.10 | 0.05 | **0.65** |

Higher values = stronger attention. "river" gets high attention from "bank".

---

## 4. Multi-Head Attention

Instead of one attention pattern, the model runs several attention "heads" in parallel — each learning a different kind of relationship.

**How it works:**
```
d_model = 512
8 heads × 64 dims each = 512
→ Concatenate all heads → Linear projection → 512
```

**What different heads learn:**

| Head | Role | Example |
|------|------|---------|
| Head 1 | Syntactic structure | Subject → verb links |
| Head 2 | Coreference | "he" → "John" |
| Head 3 | Semantic proximity | Related concepts |
| Head 4 | Long-range dependencies | Clause-level context |
| … | … | … |

**Key insight:** Multi-head attention gives the model multiple "perspectives" simultaneously — like seeing a sentence through 8 different lenses at once.

---

## 5. Positional Encoding

Attention itself has no notion of order — `"cat sat the"` would look the same as `"the cat sat"` without position information. Positional encodings are added to embeddings to inject position.

**Sinusoidal formula (original transformer):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each position gets a unique pattern of sine/cosine waves at different frequencies, added element-wise to the token embedding.

**Modern alternative — RoPE (Rotary Position Embeddings):**
- Used in LLaMA, Mistral, Gemini, and most modern LLMs
- Encodes *relative* positions rather than absolute
- Allows generalization to longer sequences than seen during training

---

## 6. Encoder vs Decoder

| | Encoder only | Decoder only | Encoder + Decoder |
|---|---|---|---|
| **Attention direction** | Bidirectional (all tokens see all) | Causal (tokens only see past) | Both |
| **Best for** | Classification, embeddings, NER | Text generation, chat, code | Translation, summarization |
| **Examples** | BERT, RoBERTa, DeBERTa | GPT-4, Claude, LLaMA, Gemini | T5, BART |

**Causal masking** (decoder): Position `i` can only attend to positions `0…i`. This is enforced by setting future attention weights to `-∞` before softmax.

**Key insight:** Most modern LLMs are decoder-only because:
1. They scale better for generation tasks
2. Pretraining on "predict the next token" is simple and scales well
3. Bidirectional context isn't needed when generating left-to-right

---

## 7. Autoregressive Generation

The model generates one token at a time. Each new token is fed back in as part of the input for the next step. This loop continues until an end-of-sequence token or max length is reached.

**Example step-through:**
```
Input:   "The capital of France is"
Step 1:  → predicts "Paris"   → input is now "The capital of France is Paris"
Step 2:  → predicts "."       → input is now "The capital of France is Paris."
Step 3:  → predicts <EOS>     → generation stops
```

**Key optimization — KV Cache:**
- At each step, key/value pairs for all past tokens are cached
- The model only computes attention for the *new* token
- Without KV cache, inference would be O(n²) per step

---

## 8. Temperature & Sampling

After computing a probability distribution over the vocabulary, the model samples the next token. Several parameters control this.

### Temperature
Scales the logits before softmax:
```
P(token) = softmax(logits / temperature)
```

| Temperature | Effect |
|-------------|--------|
| 0.0 | Always picks the highest probability token (deterministic / greedy) |
| 0.7 | Slightly creative, focused |
| 1.0 | Sample proportionally to raw probabilities |
| 1.5+ | More random, creative, sometimes incoherent |

### Top-p (Nucleus Sampling)
Sort tokens by probability, take the smallest set that sums to `p`, sample only from that set.

```
top_p = 0.9 → include tokens until cumulative probability ≥ 0.9
```

Cuts off the "long tail" of unlikely tokens dynamically.

### Top-k
Only consider the top k most probable tokens. Simpler than top-p but less adaptive.

**Example — "The capital of France is ___" at different temperatures:**

| Token | Raw prob | Temp=0.5 | Temp=1.0 | Temp=2.0 |
|-------|----------|----------|----------|----------|
| Paris | 65% | 89% | 65% | 42% |
| Lyon  | 15% | 7%  | 15% | 18% |
| Berlin| 8%  | 2%  | 8%  | 12% |
| London| 7%  | 1%  | 7%  | 12% |
| Rome  | 5%  | 1%  | 5%  | 16% |

---

## Summary: The Full Forward Pass

```
Input text
    ↓
Tokenization  →  ["The", "cat", "sat", ...]  →  [464, 2885, 3332, ...]
    ↓
Embedding lookup  →  [vector₀, vector₁, vector₂, ...]
    ↓
+ Positional encoding  →  add position info to each vector
    ↓
Transformer layers (repeated N times):
    ├── Multi-head self-attention  →  tokens exchange context
    ├── Add & LayerNorm
    ├── Feed-forward network  →  per-token transformation
    └── Add & LayerNorm
    ↓
Final linear + softmax  →  probability over vocabulary
    ↓
Sample next token  →  (temperature, top-p, top-k)
    ↓
Repeat (autoregressive loop)
```

---

## Recommended Resources

- **"Attention Is All You Need"** — Vaswani et al. (2017) — the original transformer paper
- **The Illustrated Transformer** — Jay Alammar — best visual walkthrough
- **Andrej Karpathy's "Let's build GPT"** — build a GPT from scratch in ~1 hour of video
- **3Blue1Brown Transformer series** — excellent visual intuition

---

*Phase 1 complete → Next: Phase 2 — Basic Vendor APIs*
