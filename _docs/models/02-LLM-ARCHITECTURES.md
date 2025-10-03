# LLM Architectures - Complete Deep Dive

**Zweck:** Alles über Large Language Models für RAG Generation-Phase
**Scope:** Architekturen, Model-Familien, Serving, Tools, Quantization, Production
**Zielgruppe:** Entwickler die verstehen wollen wie LLMs funktionieren und wie man sie für RAG nutzt

---

## 📋 Table of Contents

1. [Fundamentals](#1-fundamentals)
2. [Transformer Architecture](#2-transformer-architecture)
3. [Model Families](#3-model-families)
4. [Model Serving & Inference](#4-model-serving--inference)
5. [Quantization](#5-quantization)
6. [Context Window & Memory](#6-context-window--memory)
7. [Function Calling & Tools](#7-function-calling--tools)
8. [Fine-Tuning](#8-fine-tuning)
9. [Production Considerations](#9-production-considerations)
10. [Model Comparison](#10-model-comparison)

---

## 1. Fundamentals

### 1.1 Was sind Large Language Models?

**Definition:**
> LLMs sind neuronale Netze die auf riesigen Text-Korpora trainiert wurden um natürliche Sprache zu verstehen und zu generieren.

**Key Capabilities:**
1. **Text Generation** - Sätze, Absätze, ganze Dokumente schreiben
2. **Chat/Conversation** - Dialog führen, Kontext behalten
3. **Reasoning** - Logische Schlüsse ziehen, Probleme lösen
4. **Code Generation** - Programmcode schreiben und verstehen
5. **Translation** - Zwischen Sprachen übersetzen
6. **Summarization** - Texte zusammenfassen
7. **Question Answering** - Fragen beantworten (RAG!)

---

### 1.2 Größenordnungen

**Parameter Count:**
```
Small:   1-7B   (Phi-3, Gemma 2-9B)
Medium:  7-13B  (Llama 3.1-8B, Mistral 7B)
Large:   13-70B (Llama 3.1-70B, Mixtral 8x7B)
XLarge:  70B+   (Llama 3.1-405B, GPT-4, Claude)
```

**VRAM Requirements (ohne Quantization):**
```
1B Parameter  ≈ 2GB VRAM  (FP16)
7B Parameter  ≈ 14GB VRAM
13B Parameter ≈ 26GB VRAM
70B Parameter ≈ 140GB VRAM

Mit 4-bit Quantization:
7B  → 3.5GB
13B → 6.5GB
70B → 35GB
```

**Faustregel:**
```
FP16:  Parameter × 2 bytes
INT8:  Parameter × 1 byte
INT4:  Parameter × 0.5 bytes
```

---

### 1.3 Decoder-Only vs Encoder-Decoder

**Decoder-Only (GPT-Style):**
```
Input: "Die Hauptstadt von Deutschland ist"
  ↓
Decoder (autoregressive)
  ↓
Output: "Berlin"

Architecture: Nur Decoder-Blocks
Training: Next-Token-Prediction (links → rechts)
```

**Beispiele:**
- GPT (OpenAI)
- Llama (Meta)
- Mistral
- Gemma
- Claude

**Vorteile:**
- ✅ Einfacher zu trainieren
- ✅ Gut für Generation
- ✅ Flexibel einsetzbar

**Nachteile:**
- ❌ Nur unidirektional (links → rechts)

---

**Encoder-Decoder (T5-Style):**
```
Input: "Translate to German: Hello"
  ↓
Encoder (bidirectional)
  ↓
Decoder (autoregressive)
  ↓
Output: "Hallo"

Architecture: Encoder + Decoder
Training: Verschiedene Tasks (Translation, Summarization, etc.)
```

**Beispiele:**
- T5 (Google)
- BART (Facebook)
- Flan-T5

**Vorteile:**
- ✅ Bidirektionaler Encoder (besseres Verständnis)
- ✅ Gut für Task-spezifische Anwendungen

**Nachteile:**
- ❌ Komplexer
- ❌ Weniger flexibel als Decoder-Only

**Für RAG:**
> **Decoder-Only Models sind Standard** (Llama, Mistral, GPT)

---

### 1.4 Training Stages

**Stage 1: Pre-Training**
```
Daten: Wikipedia, Books, Code, Web (Billionen Tokens)
Task: Next-Token-Prediction
Dauer: Wochen/Monate auf 1000+ GPUs
Kosten: Millionen $$$
```

**Output:** Base Model (gut in Token-Prediction, aber nicht gesprächig)

---

**Stage 2: Instruction Fine-Tuning (SFT)**
```
Daten: Instruktionen + Antworten (10k-100k Beispiele)
Format:
  User: "Erkläre Photosynthese"
  Assistant: "Photosynthese ist der Prozess..."

Task: Folge Instruktionen
Dauer: Stunden/Tage
```

**Output:** Instruction-Following Model (kann Aufgaben ausführen)

---

**Stage 3: RLHF (Reinforcement Learning from Human Feedback)**
```
Daten: Human Preferences
  - Zwei Antworten generieren
  - Mensch wählt bessere aus
  - Model lernt Präferenzen

Task: Alignment (hilfsam, ehrlich, harmlos)
Dauer: Tage/Wochen
```

**Output:** Chat Model (wie ChatGPT, Claude)

**Alternative zu RLHF: DPO (Direct Preference Optimization)**
- Einfacher, schneller
- Direkt aus Präferenz-Daten lernen
- Kein Reward Model nötig

---

## 2. Transformer Architecture

### 2.1 Transformer Basics

**High-Level:**
```
Input Tokens
  ↓
Embedding Layer (Token → Vektor)
  ↓
Positional Encoding (Position im Satz)
  ↓
Transformer Blocks × N
  ├─ Multi-Head Attention
  ├─ Feed-Forward Network
  └─ Layer Normalization
  ↓
Output Layer (Vektor → Token-Probabilities)
  ↓
Next Token
```

---

### 2.2 Self-Attention Mechanism

**Konzept:**
> Jedes Token "schaut" auf alle anderen Tokens um Kontext zu verstehen

**Beispiel:**
```
Satz: "Der Hund bellt, weil er hungrig ist"

Token "er":
  - Attention zu "Hund": 0.8 ✅ (stark)
  - Attention zu "bellt": 0.3
  - Attention zu "hungrig": 0.6

→ "er" = "Hund" (Koreeferenz aufgelöst)
```

**Mathematik (vereinfacht):**
```python
# Q = Query, K = Key, V = Value
attention_scores = softmax(Q @ K.T / sqrt(d_k))
output = attention_scores @ V

# Multi-Head: Mehrere Attention-Mechanismen parallel
# → Verschiedene Aspekte gleichzeitig erfassen
```

**Komplexität:**
```
O(n²) - quadratisch in Sequenzlänge!

Beispiel:
- 512 Tokens:  262k Operationen
- 2048 Tokens: 4.2M Operationen (16x mehr!)
- 128k Tokens: 16B Operationen

→ Lange Kontexte = sehr teuer
```

---

### 2.3 Optimierungen für lange Kontexte

**Flash Attention**
```
Standard Attention: O(n²) Memory
Flash Attention:    O(n) Memory

Trick: Attention in Tiles berechnen (statt alles auf einmal)
→ 2-4x schneller, 10x weniger Memory
```

**Implementiert in:**
- vLLM
- Text Generation Inference (TGI)
- llama.cpp (teilweise)

---

**Multi-Query Attention (MQA)**
```
Standard: Alle Heads haben eigene K, V
MQA:      Alle Heads teilen K, V

Vorteil:  Weniger KV-Cache → schneller
Nachteil: Leicht schlechtere Quality
```

**Genutzt von:**
- Mistral 7B
- Falcon

---

**Grouped-Query Attention (GQA)**
```
Kompromiss zwischen Multi-Head und Multi-Query

Beispiel:
- 32 Query Heads
- 8 KV Heads (4 Queries pro KV)

Vorteil: Balance zwischen Speed und Quality
```

**Genutzt von:**
- Llama 3
- Mixtral

---

**Sliding Window Attention**
```
Statt alle Tokens anzuschauen: Nur lokales Fenster

Beispiel (Mistral):
- Window Size: 4096 Tokens
- Token 10000 schaut nur auf Token 6000-10000

Vorteil:  O(n × window_size) statt O(n²)
Nachteil: Verliert globalen Kontext (aber: durch Layers propagiert)
```

---

### 2.4 Feed-Forward Network (FFN)

**Aufbau:**
```python
# In jedem Transformer Block:
x = LayerNorm(x)
x = x + Attention(x)      # Residual Connection
x = LayerNorm(x)
x = x + FFN(x)            # Residual Connection

# FFN:
def FFN(x):
    hidden = Linear1(x)    # d_model → 4×d_model (Expansion)
    hidden = GELU(hidden)  # Activation
    output = Linear2(hidden) # 4×d_model → d_model
    return output
```

**Parameter Count:**
```
Attention: ~40% der Parameter
FFN:       ~60% der Parameter

→ FFN ist der Haupt-Kostenfaktor!
```

---

### 2.5 Mixture of Experts (MoE)

**Konzept:**
> Statt einem großen FFN: Mehrere kleine "Experten", nur 1-2 aktiv pro Token

**Beispiel (Mixtral 8x7B):**
```
8 Experten à 7B Parameter = 47B total
Pro Token: Nur 2 Experten aktiv = ~12B active

Vorteil:
- Quality von 47B Model
- Speed von 12B Model (nur aktive Parameter zählen)

Nachteil:
- Komplexer zu trainieren
- Alle Parameter müssen in VRAM (kein Swapping)
```

**Routing:**
```python
# Für jeden Token: Wähle Top-K Experten
router_probs = softmax(router(x))  # 8 Probabilities
top_k = topk(router_probs, k=2)    # Top-2 Experten

output = sum([expert_i(x) * prob_i for i in top_k])
```

**MoE Models:**
- **Mixtral 8x7B** (Mistral AI)
- **Mixtral 8x22B** (Mistral AI)
- **DeepSeek-V2** (MoE mit 128 Experten!)

---

## 3. Model Families

### 3.1 GPT Family (OpenAI)

**GPT-3.5-Turbo**
```
Parameter:     175B (geschätzt)
Context:       16k tokens
Strengths:     Schnell, günstig, gut für Chat
Cost:          $0.5 / 1M input tokens
API:           OpenAI API
```

**GPT-4**
```
Parameter:     1.7T (gemunkelt, MoE)
Context:       128k tokens
Strengths:     Reasoning, Code, multimodal (Vision)
Cost:          $10 / 1M input tokens (20x teurer als 3.5)
API:           OpenAI API
```

**GPT-4o (Omni)**
```
Parameter:     ?
Context:       128k tokens
Strengths:     Multimodal (Text, Image, Audio nativ)
Cost:          $2.5 / 1M input tokens (günstiger als GPT-4)
API:           OpenAI API
```

**GPT-4o-mini**
```
Parameter:     ~8B (geschätzt)
Context:       128k tokens
Strengths:     Sehr günstig, schnell, gut für einfache Tasks
Cost:          $0.15 / 1M input tokens
API:           OpenAI API
```

**Für RAG:**
- **GPT-4o-mini:** Beste Balance (günstig, schnell, gut genug) ⭐
- **GPT-4o:** Wenn Quality wichtiger als Kosten
- **GPT-3.5-Turbo:** Legacy, lieber 4o-mini nutzen

**Vorteile:**
- ✅ Beste Quality (besonders GPT-4)
- ✅ Große Context Window (128k)
- ✅ Multimodal (Vision, Audio)
- ✅ Function Calling / Tools
- ✅ JSON Mode

**Nachteile:**
- ❌ Kostenpflichtig
- ❌ Vendor Lock-in
- ❌ Daten gehen zu OpenAI
- ❌ Rate Limits

---

### 3.2 Claude Family (Anthropic)

**Claude 3.5 Sonnet** ⭐
```
Parameter:     ? (nicht öffentlich)
Context:       200k tokens
Strengths:     Code, Reasoning, lange Dokumente
Cost:          $3 / 1M input tokens
API:           Anthropic API
```

**Claude 3.5 Haiku**
```
Context:       200k tokens
Strengths:     Schnell, günstig
Cost:          $1 / 1M input tokens
API:           Anthropic API
```

**Claude 3 Opus**
```
Context:       200k tokens
Strengths:     Beste Quality (heavy reasoning)
Cost:          $15 / 1M input tokens
API:           Anthropic API
```

**Für RAG:**
- **Claude 3.5 Sonnet:** Top-Wahl für komplexe Queries ⭐
- **Claude 3.5 Haiku:** Schnell + günstig für einfache Antworten

**Vorteile:**
- ✅ Exzellente Quality (besser als GPT-4 für viele Tasks)
- ✅ 200k Context (mehr als GPT)
- ✅ Sehr gut für Code und lange Dokumente
- ✅ Gutes "Constitutional AI" (weniger Halluzinationen)

**Nachteile:**
- ❌ Kostenpflichtig
- ❌ Keine Vision (außer Opus/Sonnet)
- ❌ Kleineres Ökosystem als OpenAI

---

### 3.3 Llama Family (Meta)

**Llama 3.1 - 8B**
```
Parameter:     8B
Context:       128k tokens
Training:      15T tokens
Strengths:     Gut für Consumer-Hardware, schnell
License:       Open Source (kommerziell nutzbar)
Available:     Ollama, HuggingFace
```

**Llama 3.1 - 70B**
```
Parameter:     70B
Context:       128k tokens
Strengths:     Sehr gute Quality, kompetitiv zu GPT-3.5
VRAM:          ~40GB (4-bit quant)
```

**Llama 3.1 - 405B**
```
Parameter:     405B
Context:       128k tokens
Strengths:     Open-Source Alternative zu GPT-4
VRAM:          ~200GB (4-bit quant) → Multi-GPU nötig
```

**Llama 3.2 (neu, 2024)**
```
Varianten:     1B, 3B (Text-only)
               11B, 90B (Multimodal - Vision)
Context:       128k tokens
Strengths:     Kleine Models für Edge/Mobile
```

**Für RAG:**
- **Llama 3.1 - 8B:** Lokal, schnell, gut genug für viele Use Cases ⭐
- **Llama 3.1 - 70B:** Wenn beste Open-Source Quality nötig

**Vorteile:**
- ✅ Open Source
- ✅ Lokal ausführbar
- ✅ Keine API-Kosten
- ✅ 128k Context
- ✅ Viele Forks/Fine-Tunes verfügbar

**Nachteile:**
- ❌ Braucht GPU (8B: 6GB+, 70B: 40GB+)
- ❌ Etwas schlechter als GPT-4 / Claude
- ❌ Keine native Tools (aber möglich über Prompting)

---

### 3.4 Mistral Family

**Mistral 7B v0.3**
```
Parameter:     7B
Context:       32k tokens (v0.3)
Strengths:     Sehr effizient (besser als Llama 2 13B!)
License:       Apache 2.0 (Open Source)
Available:     Ollama, HuggingFace
```

**Mixtral 8x7B** (Mixture of Experts)
```
Parameter:     47B total, 12B active
Context:       32k tokens
Strengths:     Quality von 47B, Speed von 12B
VRAM:          ~30GB (4-bit, alle Parameter im VRAM!)
```

**Mixtral 8x22B**
```
Parameter:     141B total, ~40B active
Context:       64k tokens
Strengths:     Sehr gute Quality, kompetitiv zu GPT-3.5 Turbo
```

**Mistral Large 2** (API)
```
Parameter:     123B
Context:       128k tokens
Strengths:     Kompetitiv zu GPT-4
Cost:          €3 / 1M tokens (Mistral API)
```

**Für RAG:**
- **Mistral 7B:** Schnell, lokal, gut für einfache RAG ⭐
- **Mixtral 8x7B:** Beste Balance Quality/Speed (lokal)
- **Mistral Large:** API-Alternative zu GPT-4

**Vorteile:**
- ✅ Sehr effizient (beste Performance/Parameter)
- ✅ Open Source (außer Large)
- ✅ Sliding Window Attention (gute lange Kontexte)
- ✅ Function Calling Support

**Nachteile:**
- ❌ Mixtral: Alle Parameter im VRAM (kein Swapping)
- ❌ Weniger gut dokumentiert als Llama

---

### 3.5 Gemma Family (Google)

**Gemma 2 - 9B**
```
Parameter:     9B
Context:       8k tokens
Strengths:     Sehr schnell, effizient
License:       Gemma License (kommerziell nutzbar)
Available:     Ollama, HuggingFace
```

**Gemma 2 - 27B**
```
Parameter:     27B
Context:       8k tokens
Strengths:     Gute Quality, kompetitiv zu Llama 3 70B (!)
VRAM:          ~16GB (4-bit)
```

**Für RAG:**
- **Gemma 2 - 9B:** Sehr schnell, gut für Production ⭐
- **Gemma 2 - 27B:** Beste Quality für die Größe

**Vorteile:**
- ✅ Sehr schnell (optimierte Architektur)
- ✅ Gut für Instruktionen
- ✅ Kommerziell nutzbar

**Nachteile:**
- ❌ Kleiner Context (8k vs 128k bei Llama)
- ❌ Weniger Varianten als Llama/Mistral

---

### 3.6 Phi Family (Microsoft)

**Phi-3 Mini (3.8B)**
```
Parameter:     3.8B
Context:       128k tokens (!)
Strengths:     Klein aber stark, gut für Edge
VRAM:          ~2GB (4-bit)
```

**Phi-3 Small (7B)**
```
Parameter:     7B
Context:       128k tokens
```

**Phi-3 Medium (14B)**
```
Parameter:     14B
Context:       128k tokens
Strengths:     Kompetitiv zu größeren Models
```

**Für RAG:**
- **Phi-3 Mini:** Edge/Mobile RAG ⭐
- **Phi-3 Medium:** Gute Quality bei geringer Größe

**Vorteile:**
- ✅ Sehr klein + sehr gut
- ✅ 128k Context (beeindruckend für 3.8B!)
- ✅ Gut für Edge Devices

**Nachteile:**
- ❌ Weniger getestet als Llama/Mistral
- ❌ Kleineres Ökosystem

---

### 3.7 Gemini (Google, API)

**Gemini 1.5 Pro**
```
Parameter:     ? (nicht öffentlich)
Context:       1M tokens (!) - 2M experimental
Strengths:     Extrem langer Context, Multimodal (Text, Image, Video, Audio)
Cost:          $1.25 / 1M input tokens (<128k), $2.5 (128k-1M)
API:           Google AI Studio / Vertex AI
```

**Gemini 1.5 Flash**
```
Context:       1M tokens
Strengths:     Schnell, günstig
Cost:          $0.075 / 1M input tokens (<128k)
```

**Für RAG:**
- **Gemini 1.5 Flash:** Extrem günstig, langer Context ⭐
- **Gemini 1.5 Pro:** Beste für Multimodal RAG (PDFs mit Bildern/Videos)

**Vorteile:**
- ✅ 1M+ Token Context (ganzes Buch als Context!)
- ✅ Native Multimodal (besser als GPT-4V)
- ✅ Sehr günstig (Flash)

**Nachteile:**
- ❌ API-abhängig
- ❌ Weniger gut dokumentiert als OpenAI

---

## 4. Model Serving & Inference

### 4.1 Ollama

**Was ist Ollama?**
> Einfachstes Tool um LLMs lokal zu betreiben

**Installation:**
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download von ollama.com

# Verify
ollama --version
```

**Models herunterladen:**
```bash
# List verfügbare Models
ollama list

# Pull Model
ollama pull llama3.1:8b        # Llama 3.1 8B
ollama pull mistral:7b         # Mistral 7B
ollama pull gemma2:9b          # Gemma 2 9B
ollama pull phi3:mini          # Phi-3 Mini

# Embedding Models
ollama pull nomic-embed-text   # 768 dim
ollama pull mxbai-embed-large  # 1024 dim
```

**Run Model (CLI):**
```bash
# Interactive Chat
ollama run llama3.1:8b

# One-shot
ollama run llama3.1:8b "Was ist die Hauptstadt von Deutschland?"
```

**Python API:**
```python
import ollama

# Chat
response = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'system', 'content': 'Du bist ein hilfreicher Assistent.'},
        {'role': 'user', 'content': 'Was ist RAG?'}
    ]
)
print(response['message']['content'])

# Streaming
stream = ollama.chat(
    model='llama3.1:8b',
    messages=[{'role': 'user', 'content': 'Zähle bis 10'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Embeddings
emb_response = ollama.embeddings(
    model='nomic-embed-text',
    prompt='Hallo Welt'
)
embedding = emb_response['embedding']  # 768-dim vector
```

**REST API:**
```bash
# Server läuft auf http://localhost:11434

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Hallo"}]
}'

# Embeddings
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Hallo Welt"
}'
```

**Model Management:**
```bash
# List local models
ollama list

# Remove model
ollama rm llama3.1:8b

# Show model info
ollama show llama3.1:8b

# Copy model (für Custom Models)
ollama cp llama3.1:8b my-custom-model
```

**Modelfile (Custom Models):**
```bash
# Erstelle Modelfile
cat > Modelfile <<EOF
FROM llama3.1:8b

# System Prompt
SYSTEM Du bist ein Experte für Laborkühlschränke.

# Temperature
PARAMETER temperature 0.7

# Context Window
PARAMETER num_ctx 4096
EOF

# Build Model
ollama create lab-expert -f Modelfile

# Run
ollama run lab-expert "Was ist ein Laborkühlschrank?"
```

**Vorteile:**
- ✅ Extrem einfach (ein Command)
- ✅ Viele Models verfügbar (ollama.com/library)
- ✅ Automatisches Model Management
- ✅ REST API + Python Library
- ✅ Systemd Integration (läuft im Hintergrund)

**Nachteile:**
- ❌ Weniger Optimierung als vLLM/TGI
- ❌ Kein Batching (ein Request nach dem anderen)
- ❌ Kein Multi-GPU Support

**Für RAG:**
- ✅ **Perfekt für Development & kleine Deployments** ⭐
- ❌ Nicht optimal für Production mit vielen Requests

---

### 4.2 vLLM

**Was ist vLLM?**
> Production-grade LLM Serving mit PagedAttention für maximalen Throughput

**Key Innovation: PagedAttention**
```
Problem: KV-Cache fragmentiert Memory
  - Request 1: 512 tokens  → 2GB KV-Cache
  - Request 2: 128 tokens  → 0.5GB KV-Cache
  - Request 3: 2048 tokens → 8GB KV-Cache
  → Viel ungenutzter Speicher zwischen Blocks

Lösung: Paging (wie OS Virtual Memory)
  - KV-Cache in kleine Blocks aufteilen (z.B. 16 tokens)
  - Nur genutzte Blocks allokieren
  → 2-4x mehr Requests gleichzeitig
```

**Installation:**
```bash
pip install vllm

# Oder mit CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

**Python API:**
```python
from vllm import LLM, SamplingParams

# Model laden
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1  # Anzahl GPUs
)

# Sampling Parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# Single Request
outputs = llm.generate(
    "Was ist die Hauptstadt von Deutschland?",
    sampling_params
)
print(outputs[0].outputs[0].text)

# Batch Requests (effizient!)
prompts = [
    "Was ist RAG?",
    "Erkläre Photosynthese",
    "Was ist die Quadratwurzel von 144?"
]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

**OpenAI-Compatible Server:**
```bash
# Server starten
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1

# Client (OpenAI Library nutzen!)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM braucht keinen Key
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Was ist RAG?"}
    ]
)
print(response.choices[0].message.content)
```

**Advanced Features:**
```python
# Multi-GPU (Tensor Parallelism)
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=2  # 2 GPUs
)

# Quantization
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization="awq"  # AWQ 4-bit
)

# GPU Memory Utilization
llm = LLM(
    model="...",
    gpu_memory_utilization=0.9  # 90% VRAM nutzen (default: 0.9)
)
```

**Benchmarks:**
```
Throughput (requests/second):
- Ollama:  ~2-3 req/s (sequential)
- vLLM:    ~20-30 req/s (batched, PagedAttention)

→ 10x mehr Throughput!
```

**Vorteile:**
- ✅ Höchster Throughput (PagedAttention)
- ✅ OpenAI-compatible API
- ✅ Batching (mehrere Requests gleichzeitig)
- ✅ Multi-GPU Support
- ✅ Quantization Support (AWQ, GPTQ)

**Nachteile:**
- ❌ Komplexer Setup als Ollama
- ❌ Keine Model Management (Models manuell laden)
- ❌ CUDA-only (keine CPU-Inferenz)

**Für RAG:**
- ✅ **Beste Wahl für Production** ⭐

---

### 4.3 Text Generation Inference (TGI, Hugging Face)

**Was ist TGI?**
> Production Serving für Hugging Face Models mit Tensor Parallelism

**Docker Setup:**
```bash
# GPU
docker run --gpus all \
  -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-input-length 4096 \
  --max-total-tokens 8192

# Multi-GPU
docker run --gpus all \
  ... \
  --num-shard 2  # 2 GPUs
```

**Python Client:**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

# Chat
response = client.text_generation(
    "Was ist RAG?",
    max_new_tokens=512,
    temperature=0.7
)
print(response)

# Streaming
for token in client.text_generation(
    "Zähle bis 10",
    max_new_tokens=100,
    stream=True
):
    print(token, end='', flush=True)
```

**Features:**
- ✅ Tensor Parallelism (Multi-GPU)
- ✅ Flash Attention
- ✅ Paged Attention
- ✅ Quantization (bitsandbytes, AWQ, GPTQ)
- ✅ OpenTelemetry (Monitoring)

**vs vLLM:**
- TGI: Besser für Hugging Face Ökosystem
- vLLM: Höherer Throughput
- Beide: Production-grade

---

### 4.4 llama.cpp

**Was ist llama.cpp?**
> Pure C++ Implementierung für CPU-Inferenz (Basis für Ollama)

**Installation:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Mit CUDA
make LLAMA_CUBLAS=1

# Mit Metal (Mac M1/M2/M3)
make LLAMA_METAL=1
```

**Model Download (GGUF Format):**
```bash
# Von HuggingFace (TheBloke hat viele GGUF Models)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

**CLI:**
```bash
./main \
  -m llama-2-7b-chat.Q4_K_M.gguf \
  -p "Was ist RAG?" \
  -n 512 \
  --temp 0.7
```

**Server:**
```bash
./server \
  -m llama-2-7b-chat.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096  # context size

# OpenAI-compatible API auf http://localhost:8080
```

**Python (via llama-cpp-python):**
```bash
pip install llama-cpp-python

# Mit CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,      # context window
    n_threads=8,     # CPU threads
    n_gpu_layers=32  # Layers auf GPU (0 = CPU only)
)

output = llm(
    "Was ist RAG?",
    max_tokens=512,
    temperature=0.7
)
print(output['choices'][0]['text'])
```

**Vorteile:**
- ✅ CPU-Inferenz (kein CUDA nötig)
- ✅ Sehr effizient (optimierter C++ Code)
- ✅ Apple Silicon Support (Metal)
- ✅ Kleine Binaries (~1MB)

**Nachteile:**
- ❌ Langsamer als GPU (aber gut für CPU!)
- ❌ GGUF-Format erforderlich

**Für RAG:**
- ✅ Wenn keine GPU verfügbar
- ✅ Edge Deployments (Raspberry Pi, etc.)

---

### 4.5 Zusammenfassung Serving

| Tool | Best For | Throughput | Setup | GPU |
|------|----------|------------|-------|-----|
| **Ollama** | Development, kleine Deployments | ⚡ | ⭐⭐⭐ Einfach | Optional |
| **vLLM** | Production, hoher Traffic | ⚡⚡⚡ | ⭐⭐ Medium | Erforderlich |
| **TGI** | Hugging Face Ecosystem | ⚡⚡⚡ | ⭐ Komplex | Erforderlich |
| **llama.cpp** | CPU/Edge Deployments | ⚡ | ⭐⭐ Medium | Optional |

**Empfehlung für RAG:**
```
Development:     Ollama
Production:      vLLM (GPU) oder Ollama (kleine Last)
Edge/CPU:        llama.cpp
Cloud:           API (OpenAI, Claude, Gemini)
```

---

## 5. Quantization

### 5.1 Was ist Quantization?

**Konzept:**
> Reduziere Präzision der Gewichte um Speicher/Speed zu sparen

**Float Precision:**
```
FP32 (Full Precision):     4 bytes pro Gewicht
FP16 (Half Precision):     2 bytes
BF16 (Brain Float):        2 bytes (besserer Range als FP16)
INT8 (8-bit Integer):      1 byte
INT4 (4-bit Integer):      0.5 bytes
```

**Beispiel (Llama 3.1 8B):**
```
FP16:  8B × 2 bytes = 16GB
INT8:  8B × 1 byte  = 8GB   (50% Reduktion)
INT4:  8B × 0.5 bytes = 4GB (75% Reduktion)
```

---

### 5.2 GGUF (llama.cpp Format)

**Was ist GGUF?**
> GPT-Generated Unified Format - Standard für llama.cpp/Ollama

**Quantization Schemes:**
```
Q2_K:   2-bit (sehr klein, schlechte Quality)
Q3_K_S: 3-bit Small
Q3_K_M: 3-bit Medium
Q4_0:   4-bit (alt, nicht empfohlen)
Q4_K_S: 4-bit K-Quant Small ⭐ (gut für CPU)
Q4_K_M: 4-bit K-Quant Medium ⭐ (Standard)
Q5_K_S: 5-bit K-Quant Small
Q5_K_M: 5-bit K-Quant Medium ⭐ (beste Balance)
Q6_K:   6-bit
Q8_0:   8-bit (fast keine Quality-Loss)
```

**Namensschema:**
```
llama-3.1-8b-instruct.Q4_K_M.gguf
                      ^^^^^^
                      Quantization
```

**K-Quant:**
- K = Kalman (statistisch optimiert)
- Verschiedene Quantization pro Layer-Typ
- Besser als naive Quantization

**Quality vs Size:**
```
Q4_K_M (4-bit): 4.5GB   - Gut genug für die meisten Use Cases ⭐
Q5_K_M (5-bit): 5.5GB   - Bessere Quality, etwas größer
Q8_0   (8-bit): 8.5GB   - Fast kein Quality-Loss
FP16:           16GB    - Original
```

**Empfehlung:**
- **Q4_K_M:** Standard für die meisten Use Cases
- **Q5_K_M:** Wenn genug VRAM (bessere Quality)
- **Q8_0:** Wenn maximale Quality (wenig Benefit vs FP16)

---

### 5.3 GPTQ (GPU-optimized)

**Was ist GPTQ?**
> Quantization optimiert für GPU-Inferenz

**Wie funktioniert's:**
1. Calibration: Nutze Datensample um Quantization-Parameter zu finden
2. Gewichte in INT4 konvertieren
3. Activations bleiben FP16

**Models:**
```
TheBloke auf HuggingFace hat viele GPTQ Models:
- TheBloke/Llama-2-7B-Chat-GPTQ
- TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
```

**Laden:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GPTQ",
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")
```

**Vorteile:**
- ✅ Sehr schnell auf GPU
- ✅ 4-bit → 75% weniger VRAM
- ✅ ~1-2% Quality-Loss

**Nachteile:**
- ❌ GPU-only (nicht für CPU)
- ❌ Calibration nötig (dauert)

---

### 5.4 AWQ (Activation-aware Weight Quantization)

**Was ist AWQ?**
> Noch bessere Quantization als GPTQ - berücksichtigt Activations

**Key Idea:**
```
Nicht alle Gewichte sind gleich wichtig!
→ Schütze wichtige Gewichte vor Quantization
→ Aggressivere Quantization für unwichtige Gewichte
```

**Performance:**
```
Perplexity (lower = better):
FP16:   5.47
GPTQ:   5.54 (+1.3%)
AWQ:    5.50 (+0.5%) ✅ Besser als GPTQ
```

**Laden (vLLM):**
```python
from vllm import LLM

llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq"
)
```

**Vorteile:**
- ✅ Bessere Quality als GPTQ
- ✅ Gleicher Speicher (4-bit)
- ✅ Etwas schneller als GPTQ

**Nachteile:**
- ❌ Weniger weit verbreitet als GPTQ
- ❌ GPU-only

**Empfehlung:**
> Wenn verfügbar: AWQ > GPTQ

---

### 5.5 bitsandbytes (Hugging Face)

**Was ist bitsandbytes?**
> 8-bit und 4-bit Quantization für Hugging Face Models

**8-bit (LLM.int8()):**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,    # 8-bit Quantization
    device_map="auto"
)

# 7B Model: 14GB → 7GB ✅
```

**4-bit (QLoRA):**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat 4-bit
    bnb_4bit_use_double_quant=True, # Nested Quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 7B Model: 14GB → 3.5GB ✅
```

**Vorteile:**
- ✅ Einfach (ein Parameter)
- ✅ On-the-fly (kein Pre-Quantization nötig)
- ✅ Gut für Fine-Tuning (QLoRA)

**Nachteile:**
- ❌ Etwas langsamer als GPTQ/AWQ
- ❌ Nur für Inference + Training, nicht optimal für Production

---

### 5.6 Quantization Comparison

| Format | Precision | VRAM (7B) | Speed | Quality | Use Case |
|--------|-----------|-----------|-------|---------|----------|
| **FP16** | 16-bit | 14GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Baseline |
| **GGUF Q4_K_M** | 4-bit | 4.5GB | ⚡⚡ (CPU) | ⭐⭐⭐⭐ | Ollama, CPU |
| **GGUF Q5_K_M** | 5-bit | 5.5GB | ⚡⚡ (CPU) | ⭐⭐⭐⭐ | Bessere Quality |
| **GPTQ** | 4-bit | 4.5GB | ⚡⚡⚡ (GPU) | ⭐⭐⭐⭐ | GPU Production |
| **AWQ** | 4-bit | 4.5GB | ⚡⚡⚡ (GPU) | ⭐⭐⭐⭐ | GPU Production (beste) |
| **bitsandbytes 4-bit** | 4-bit | 3.5GB | ⚡⚡ | ⭐⭐⭐⭐ | Fine-Tuning |

**Empfehlung:**
```
Ollama/CPU:      GGUF Q4_K_M oder Q5_K_M
vLLM/GPU:        AWQ (wenn verfügbar) sonst GPTQ
Fine-Tuning:     bitsandbytes 4-bit (QLoRA)
Beste Quality:   FP16 (wenn VRAM erlaubt)
```

---

## 6. Context Window & Memory

### 6.1 Was ist Context Window?

**Definition:**
> Maximale Anzahl Tokens die das Model gleichzeitig "sehen" kann

**Beispiele:**
```
GPT-3.5:       16k tokens  (~12k Wörter, ~24 Seiten)
GPT-4:         128k tokens (~96k Wörter, ~192 Seiten)
Claude 3.5:    200k tokens (~150k Wörter, ~300 Seiten)
Llama 3.1:     128k tokens
Gemini 1.5:    1M tokens   (~750k Wörter, ganzes Buch!)
```

**Warum wichtig für RAG?**
```
RAG Pipeline:
1. Retrieve Top-K Chunks (z.B. 10 Chunks à 500 Tokens = 5k Tokens)
2. System Prompt (500 Tokens)
3. User Query (100 Tokens)
4. Chat History (1k Tokens)

Total Input: ~6.5k Tokens

Response: ~500 Tokens

→ Brauchen min. 7k Context Window
```

---

### 6.2 KV-Cache

**Problem: Autoregressive Generation ist langsam**
```
Generate "Die Hauptstadt von Deutschland ist Berlin"

Step 1: Input: "Die"                     → Output: "Hauptstadt"
Step 2: Input: "Die Hauptstadt"          → Output: "von"
Step 3: Input: "Die Hauptstadt von"      → Output: "Deutschland"
...

→ Jeder Step muss alle vorherigen Tokens neu prozessieren!
```

**Lösung: KV-Cache**
```
Speichere Key & Value Matrizen aus Attention

Step 1: Compute K, V für "Die"
Step 2: Re-use K, V von "Die", compute nur für "Hauptstadt"
Step 3: Re-use K, V von "Die Hauptstadt", compute nur für "von"

→ Viel schneller!
```

**Memory Cost:**
```
KV-Cache Size = 2 × num_layers × d_model × sequence_length

Beispiel (Llama 3.1 8B, 128k context):
= 2 × 32 × 4096 × 128000 × 2 bytes (FP16)
= ~64GB (!!)

→ Lange Kontexte = sehr viel VRAM für KV-Cache
```

**Optimierungen:**
- **PagedAttention (vLLM):** Dynamische Allokation
- **Flash Attention:** Weniger Memory
- **Multi-Query Attention:** Kleinerer KV-Cache

---

### 6.3 Context Window Erweiterung

**Problem:**
> Model trainiert auf 4k Context, aber wir wollen 128k

**Techniken:**

**1. Positional Interpolation (PI)**
```
Original: Position Embeddings für [0, 4096]
Erweitert: Skaliere auf [0, 128k]

Position 128000 → 128000 / 32 = 4000 (innerhalb Original-Range)
```

**2. YaRN (Yet another RoPE extensioN)**
```
Verbesserte Positional Interpolation
→ Bessere Performance auf langen Kontexten
```

**3. ALiBi (Attention with Linear Biases)**
```
Keine Positional Embeddings!
→ Stattdessen: Linear Bias in Attention
→ Extrapoliert besser zu längeren Sequenzen
```

**Genutzt von:**
- Llama 3.1: RoPE + PI → 128k
- Mistral: Sliding Window + RoPE → 32k
- MPT: ALiBi → 65k

---

### 6.4 Long Context Best Practices für RAG

**1. Chunking optimieren**
```python
# Nicht zu große Chunks!
# Besser: 10 Chunks à 500 Tokens
# Schlechter: 50 Chunks à 100 Tokens (zu viel Overhead)

chunk_size = 500  # Tokens
top_k = 10        # Chunks
total_context = chunk_size × top_k = 5000 Tokens ✅
```

**2. Context Compression**
```python
# Statt alle Chunks vollständig:
# → Summarize oder nutze nur relevante Snippets

def compress_context(chunks, max_tokens=3000):
    compressed = []
    total = 0
    for chunk in chunks:
        if total + len(chunk) > max_tokens:
            # Kürze Chunk
            remaining = max_tokens - total
            compressed.append(chunk[:remaining])
            break
        compressed.append(chunk)
        total += len(chunk)
    return compressed
```

**3. Re-Ranking nutzen**
```python
# Retrieve 100 Chunks (schnell, unpräzise)
# Re-Rank zu Top-10 (langsam, präzise)
# → Nur Top-10 in Context
# → Spart Token, bessere Relevanz
```

**4. Streaming für lange Antworten**
```python
# Statt alles auf einmal generieren:
# → Stream Token by Token
# → User sieht sofort Ergebnis
# → Bessere UX
```

---

## 7. Function Calling & Tools

### 7.1 Was ist Function Calling?

**Konzept:**
> LLM kann externe Funktionen/Tools aufrufen

**Beispiel:**
```
User: "Was ist das Wetter in Berlin?"

LLM (ohne Tools): "Ich habe keinen Zugriff auf aktuelle Wetterdaten"

LLM (mit Tools):
1. Erkenne: Brauche Wetter-API
2. Rufe auf: get_weather(location="Berlin")
3. Erhalte: {"temp": 15, "condition": "cloudy"}
4. Antworte: "In Berlin sind es aktuell 15°C und bewölkt"
```

---

### 7.2 OpenAI Function Calling

**Tool Definition:**
```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_specs",
            "description": "Ruft technische Spezifikationen eines Produkts ab",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Die Produkt-ID (z.B. LABO-288)"
                    }
                },
                "required": ["product_id"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Was sind die Specs vom LABO-288?"}
]

# LLM Call mit Tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # LLM entscheidet ob Tool nötig
)

# Check ob Tool-Call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    print(f"LLM will call: {function_name}({function_args})")
    # → get_product_specs({"product_id": "LABO-288"})

    # Execute Tool
    result = get_product_specs(**function_args)

    # Send Result back to LLM
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Final Response
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print(final_response.choices[0].message.content)
```

---

### 7.3 Parallel Function Calling

**Multi-Tool-Call:**
```python
User: "Was kosten LABO-288 und MED-COOL-300?"

LLM generiert:
[
    tool_call_1: get_price(product_id="LABO-288"),
    tool_call_2: get_price(product_id="MED-COOL-300")
]

→ Beide parallel ausführen!
```

**Code:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    parallel_tool_calls=True  # ✅ Parallel
)

# Execute all tool calls
for tool_call in response.choices[0].message.tool_calls:
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    # Execute
    result = execute_function(function_name, function_args)

    # Add to messages
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

# Final Response mit allen Results
final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
```

---

### 7.4 Function Calling mit Open-Source LLMs

**Problem:**
> Llama, Mistral etc. haben kein natives Function Calling

**Lösung 1: Prompt Engineering**
```python
system_prompt = """
Du hast Zugriff auf folgende Tools:

get_product_specs(product_id: str) -> dict
  Beschreibung: Ruft Produktspezifikationen ab
  Beispiel: get_product_specs("LABO-288")

Um ein Tool zu nutzen, antworte im Format:
TOOL_CALL: <function_name>(<arguments>)

Beispiel:
TOOL_CALL: get_product_specs("LABO-288")
"""

user_query = "Was sind die Specs vom LABO-288?"

response = llm.generate(system_prompt + user_query)

# Parse Response
if "TOOL_CALL:" in response:
    # Extract function call
    tool_call = parse_tool_call(response)
    result = execute_tool(tool_call)

    # Send result back
    final_response = llm.generate(
        system_prompt + user_query +
        f"\nTOOL_RESULT: {result}\n\nBeantworte die Frage basierend auf dem Ergebnis."
    )
```

**Lösung 2: Fine-Tuned Models**
```
Mixtral-8x7B-Instruct-v0.1 hat Function Calling Support!
Llama 3.1 (Instruct) kann auch Tools (experimentell)

→ Nutze diese statt Base Models
```

**Beispiel (Llama 3.1 mit Tools):**
```python
# Llama 3.1 unterstützt Tool-Format (ähnlich wie OpenAI)
messages = [
    {
        "role": "system",
        "content": "You have access to these tools: ..."
    },
    {
        "role": "user",
        "content": "What are the specs of LABO-288?"
    }
]

# Ollama (mit Llama 3.1)
response = ollama.chat(
    model='llama3.1:8b',
    messages=messages,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_product_specs",
            "description": "...",
            "parameters": {...}
        }
    }]
)

# Check für Tool-Calls
if 'tool_calls' in response['message']:
    # Ähnlich wie OpenAI
    ...
```

---

### 7.5 RAG-Specific Tools

**Beispiel-Tools für RAG:**

**1. Vector Search Tool**
```python
{
    "name": "search_knowledge_base",
    "description": "Durchsucht die Produktdatenbank nach relevanten Informationen",
    "parameters": {
        "query": "string - Suchquery",
        "top_k": "integer - Anzahl Ergebnisse (default: 5)"
    }
}
```

**2. Structured Query Tool**
```python
{
    "name": "filter_products",
    "description": "Filtert Produkte nach technischen Kriterien",
    "parameters": {
        "min_volume": "integer - Minimales Volumen in Litern",
        "max_volume": "integer - Maximales Volumen in Litern",
        "category": "string - Produktkategorie",
        "manufacturer": "string - Hersteller"
    }
}
```

**3. Get Specific Product**
```python
{
    "name": "get_product_details",
    "description": "Ruft alle Details eines bestimmten Produkts ab",
    "parameters": {
        "product_id": "string - Produkt-ID"
    }
}
```

**Workflow:**
```
User: "Zeig mir Kühlschränke von Liebherr mit mehr als 250L"

LLM:
1. Tool-Call: filter_products(manufacturer="Liebherr", min_volume=250)
2. Erhalte: [LABO-288, LABO-340, ...]
3. Tool-Call: get_product_details("LABO-288")
4. Tool-Call: get_product_details("LABO-340")
5. Generiere Antwort mit Details
```

---

## 8. Fine-Tuning

### 8.1 Wann LLM Fine-Tuning?

**JA, wenn:**
- ✅ Spezifische Antwort-Style (formal, technisch, etc.)
- ✅ Domain-spezifisches Vokabular konsistent nutzen
- ✅ Bestimmte Aufgaben-Struktur einhalten
- ✅ Reduziere Halluzinationen in deiner Domain

**NEIN, wenn:**
- ❌ Base Model funktioniert gut mit Prompt Engineering
- ❌ Wenig Trainingsdaten (<1000 Beispiele)
- ❌ Keine GPU für Training

**Für RAG:**
> Fine-Tuning der Generation (LLM) ist seltener nötig als Embedding-Fine-Tuning
> → Meist reicht guter System-Prompt

---

### 8.2 LoRA (Low-Rank Adaptation)

**Konzept:**
```
Statt alle Parameter zu trainieren:
→ Trainiere kleine "Adapter" Matrizen

Original Weight: W (groß, frozen)
LoRA Update:     ΔW = A × B (klein, trainable)
Final Weight:    W' = W + α × ΔW

Dimensionen:
W:  4096 × 4096 = 16M Parameter
A:  4096 × 8    = 32k Parameter
B:  8 × 4096    = 32k Parameter
→ Nur 64k trainable (0.4% von W!)
```

**Code:**
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Base Model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_4bit=True,  # QLoRA
    device_map="auto"
)

# LoRA Config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                # Rank (höher = mehr Kapazität, mehr Parameter)
    lora_alpha=16,      # Scaling factor
    lora_dropout=0.1,
    target_modules=[    # Welche Layer mit LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: 4M (0.06% of 7B!)
```

**Training:**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-llama-rag",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# Save LoRA Adapter (nur ~10-50MB!)
model.save_pretrained("./lora-adapter")
```

**Inference:**
```python
from peft import PeftModel

# Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf"
)

# Load LoRA Adapter
model = PeftModel.from_pretrained(
    base_model,
    "./lora-adapter"
)

# Use normally
output = model.generate(...)
```

**Vorteile:**
- ✅ Sehr wenig VRAM (7B Model mit 6GB trainierbar!)
- ✅ Schnelles Training (wenige Parameter)
- ✅ Kleine Adapter-Files (einfach zu teilen)
- ✅ Mehrere Adapter für verschiedene Tasks

**Nachteile:**
- ❌ Etwas schlechter als Full Fine-Tuning (~95% Quality)
- ❌ Für sehr große Änderungen nicht ideal

---

### 8.3 QLoRA (Quantized LoRA)

**Konzept:**
> LoRA + 4-bit Base Model = Extremst speichereffizient

**Code:**
```python
from transformers import BitsAndBytesConfig

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA (same as before)
model = get_peft_model(model, lora_config)

# Train!
# 7B Model: ~6GB VRAM ✅
```

**VRAM Requirements:**
```
Llama 2 7B:
- Full Fine-Tuning:  ~28GB
- LoRA (FP16):       ~14GB
- QLoRA (4-bit):     ~6GB ✅

→ Consumer GPU (RTX 3090, 4090) ausreichend!
```

---

### 8.4 Training Data Format

**Instruction Format:**
```json
{
  "instruction": "Beantworte die Frage basierend auf dem Kontext",
  "input": "Kontext: Der LABO-288 hat 280L Volumen.\n\nFrage: Wie viel Volumen hat der LABO-288?",
  "output": "Der LABO-288 hat ein Volumen von 280 Litern."
}
```

**Chat Format (besser für RAG):**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Du bist ein Assistent für Laborkühlschränke. Beantworte Fragen basierend auf dem bereitgestellten Kontext."
    },
    {
      "role": "user",
      "content": "Kontext:\n- LABO-288: 280L, Temperaturüberwachung\n- MED-COOL-300: 300L, keine Überwachung\n\nFrage: Welcher Kühlschrank hat Temperaturüberwachung?"
    },
    {
      "role": "assistant",
      "content": "Der LABO-288 verfügt über Temperaturüberwachung."
    }
  ]
}
```

**Synthetic Data Generation:**
```python
# Nutze GPT-4 um Training-Daten zu generieren
import openai

def generate_training_example(product):
    prompt = f"""
    Generiere 3 realistische Frage-Antwort-Paare für dieses Produkt:
    {json.dumps(product, indent=2)}

    Format: JSON Array mit {{ "question": "...", "answer": "..." }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    examples = json.loads(response.choices[0].message.content)
    return examples

# Für alle Produkte
training_data = []
for product in products:
    examples = generate_training_example(product)
    training_data.extend(examples)

# → 1000+ Training-Beispiele in wenigen Minuten
```

---

## 9. Production Considerations

### 9.1 Latency Optimization

**1. Model Size**
```
GPT-4:          ~5-10s  (API, groß)
GPT-4o-mini:    ~1-2s   (API, kleiner)
Llama 3.1 70B:  ~3-5s   (lokal, 4-bit)
Llama 3.1 8B:   ~0.5-1s (lokal, 4-bit) ✅

→ Kleinere Models = schneller
```

**2. Quantization**
```
FP16:  10 tokens/sec
INT4:  40 tokens/sec

→ 4x speedup!
```

**3. Batching (vLLM)**
```
Sequential: 1 req × 2s = 2s/req
Batched:    5 reqs × 3s = 0.6s/req

→ 3x mehr Throughput
```

**4. Speculative Decoding**
```
Konzept: Kleines Model rät Token voraus, großes Model verifiziert

Draft Model (Llama 8B):  10 Tokens vorschlagen
Target Model (Llama 70B): Verifizieren (parallel!)

→ 2-3x speedup
```

**5. Caching**
```python
# System Prompt cachen (ändert sich nicht)
system_prompt = "Du bist ein Assistent für..."  # Immer gleich

# Nur User-Teil neu embedden
# → Spart Tokens, schneller
```

---

### 9.2 Cost Optimization (API)

**Token Counting:**
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text):
    return len(encoding.encode(text))

# Vor API-Call checken!
total_tokens = (
    count_tokens(system_prompt) +
    count_tokens(context) +
    count_tokens(user_query)
)

estimated_cost = total_tokens / 1_000_000 * 0.15  # $0.15/1M tokens
print(f"Estimated cost: ${estimated_cost:.4f}")
```

**Cost Reduction:**

**1. Kontext komprimieren**
```python
# Statt 10 Chunks à 500 Tokens (5000 total):
# → Summarize zu 2000 Tokens
# → 60% weniger Kosten
```

**2. Kleineres Model für einfache Queries**
```python
# Routing basierend auf Query-Komplexität
if is_simple_query(query):
    model = "gpt-4o-mini"  # $0.15/1M
else:
    model = "gpt-4o"       # $2.50/1M
```

**3. Caching (OpenAI Prompt Caching)**
```python
# OpenAI cached System Prompts automatisch
# → 50% günstiger für gecachte Tokens

# Best Practice: Stabiler System Prompt
system_prompt = """..."""  # Ändert sich nie

# Nur User-Teil variiert
user_part = f"Context: {context}\n\nQuestion: {query}"
```

**4. Streaming (bessere UX, nicht günstiger)**
```python
# Kein Cost-Vorteil, aber User sieht Ergebnis sofort
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True
):
    print(chunk.choices[0].delta.content, end='')
```

---

### 9.3 Quality Monitoring

**Metrics:**

**1. Response Quality**
```python
# Human Evaluation (Gold Standard)
# - Ist Antwort korrekt?
# - Ist Antwort hilfreich?
# - Nutzt Antwort den Kontext?

# Automatic (LLM-as-Judge)
judge_prompt = f"""
Rate die folgende Antwort auf einer Skala von 1-5:

Kontext: {context}
Frage: {query}
Antwort: {response}

Kriterien:
- Korrektheit (nutzt Kontext?)
- Vollständigkeit
- Relevanz

Score (1-5):
"""

judge_response = gpt4.generate(judge_prompt)
score = parse_score(judge_response)
```

**2. Hallucination Detection**
```python
# Check ob Antwort Infos enthält die NICHT im Kontext sind

def detect_hallucination(context, response):
    prompt = f"""
    Kontext: {context}
    Antwort: {response}

    Enthält die Antwort Informationen die NICHT im Kontext stehen?
    Antworte mit JA oder NEIN.
    """

    result = llm.generate(prompt)
    return "JA" in result

# Log hallucinations
if detect_hallucination(context, response):
    log_warning("Possible hallucination detected")
```

**3. Latency Tracking**
```python
import time

start = time.time()
response = llm.generate(prompt)
latency = time.time() - start

# Log
metrics.log({
    'latency': latency,
    'model': model_name,
    'tokens': count_tokens(prompt + response)
})

# Alert wenn zu langsam
if latency > 5.0:
    alert("High latency detected")
```

**4. Cost Tracking**
```python
# Track Kosten pro Request
def log_usage(prompt, response, model):
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response)

    cost = calculate_cost(input_tokens, output_tokens, model)

    db.insert({
        'timestamp': now(),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'cost': cost,
        'model': model
    })

# Daily Report
daily_cost = db.query("SELECT SUM(cost) FROM usage WHERE date = today()")
print(f"Today's cost: ${daily_cost:.2f}")
```

---

### 9.4 Error Handling

**API Errors:**
```python
from openai import OpenAI, RateLimitError, APIError
import time

client = OpenAI()

def generate_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                timeout=30  # Timeout nach 30s
            )
            return response.choices[0].message.content

        except RateLimitError:
            # Rate limit → warten
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limit, waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIError as e:
            # Server Error → retry
            print(f"API Error: {e}, retrying...")
            time.sleep(1)

        except Exception as e:
            # Unbekannter Error → log & fail
            log_error(f"Unexpected error: {e}")
            raise

    raise Exception("Max retries exceeded")
```

**Local Model Errors:**
```python
# Out of Memory
try:
    response = llm.generate(prompt)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size oder nutze kleineres Model
        torch.cuda.empty_cache()
        response = llm.generate(prompt, max_tokens=256)  # Kürzere Response
    else:
        raise
```

**Timeout Handling:**
```python
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Set timeout (10 seconds)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)

try:
    response = llm.generate(prompt)
    signal.alarm(0)  # Cancel alarm
except TimeoutException:
    print("Generation timed out!")
    response = "Sorry, generation took too long."
```

---

## 10. Model Comparison

### 10.1 Feature Matrix

| Model | Params | Context | Open Source | API | Function Calling | Vision | Cost (1M in) |
|-------|--------|---------|-------------|-----|------------------|--------|--------------|
| **GPT-4o** | ? | 128k | ❌ | ✅ | ✅ | ✅ | $2.50 |
| **GPT-4o-mini** | ~8B | 128k | ❌ | ✅ | ✅ | ✅ | $0.15 |
| **Claude 3.5 Sonnet** | ? | 200k | ❌ | ✅ | ✅ | ✅ | $3.00 |
| **Claude 3.5 Haiku** | ? | 200k | ❌ | ✅ | ✅ | ❌ | $1.00 |
| **Gemini 1.5 Flash** | ? | 1M | ❌ | ✅ | ✅ | ✅ | $0.075 |
| **Llama 3.1 8B** | 8B | 128k | ✅ | ❌ | ⚠️ | ❌ | Free (local) |
| **Llama 3.1 70B** | 70B | 128k | ✅ | ❌ | ⚠️ | ❌ | Free (local) |
| **Mistral 7B** | 7B | 32k | ✅ | ❌ | ⚠️ | ❌ | Free (local) |
| **Mixtral 8x7B** | 47B | 32k | ✅ | ❌ | ⚠️ | ❌ | Free (local) |
| **Gemma 2 9B** | 9B | 8k | ✅ | ❌ | ❌ | ❌ | Free (local) |
| **Phi-3 Mini** | 3.8B | 128k | ✅ | ❌ | ❌ | ❌ | Free (local) |

⚠️ = Via Prompt Engineering möglich, nicht nativ

---

### 10.2 Performance Benchmarks (MMLU, HumanEval)

**MMLU (Massive Multitask Language Understanding):**
```
GPT-4o:            88.7%
Claude 3.5 Sonnet: 88.3%
GPT-4o-mini:       82.0%
Llama 3.1 70B:     79.3%
Gemini 1.5 Flash:  78.9%
Mixtral 8x7B:      70.6%
Llama 3.1 8B:      66.7%
Mistral 7B:        62.5%
Gemma 2 9B:        71.3%
Phi-3 Mini:        68.8%
```

**HumanEval (Code Generation):**
```
GPT-4o:            90.2%
Claude 3.5 Sonnet: 92.0% ← Beste für Code!
GPT-4o-mini:       87.2%
Llama 3.1 70B:     80.5%
Gemini 1.5 Flash:  74.3%
Mixtral 8x7B:      60.7%
Llama 3.1 8B:      72.6%
Gemma 2 9B:        61.4%
```

---

### 10.3 RAG-Specific Recommendations

**Development (lokal, kostenlos):**
```
1. Llama 3.1 8B     - Beste Balance Speed/Quality ⭐
2. Mistral 7B       - Sehr effizient
3. Gemma 2 9B       - Schnell, gut für Instruktionen
4. Phi-3 Mini       - Wenn wenig VRAM (3.8B!)
```

**Production (API, Quality > Cost):**
```
1. Claude 3.5 Sonnet - Beste Quality, lange Kontexte ⭐
2. GPT-4o           - Sehr gut, Multimodal
3. GPT-4o-mini      - Gut genug, sehr günstig
```

**Production (API, Cost > Quality):**
```
1. Gemini 1.5 Flash - Extrem günstig ($0.075/1M), 1M context! ⭐
2. GPT-4o-mini      - Gute Balance
3. Claude 3.5 Haiku - Schnell, günstiger als Sonnet
```

**Production (lokal, hoher Traffic):**
```
1. Llama 3.1 8B (vLLM)  - Beste Throughput ⭐
2. Mistral 7B (vLLM)    - Sehr effizient
3. Gemma 2 9B (vLLM)    - Schneller als Llama
```

**Specialized (Code-heavy RAG):**
```
1. Claude 3.5 Sonnet    - Beste für Code ⭐
2. GPT-4o              - Sehr gut
3. Llama 3.1 70B       - Beste Open-Source für Code
```

**Specialized (Multimodal RAG - PDFs mit Bildern):**
```
1. Gemini 1.5 Pro  - Native Multimodal, 1M context ⭐
2. GPT-4o          - Sehr gut für Vision
3. Claude 3.5 Sonnet - Gut, aber teurer
```

---

## Summary: Quick Decision Guide

### Model Selection:
```
Development lokal:           Llama 3.1 8B (Ollama)
Production API (Quality):    Claude 3.5 Sonnet
Production API (Cost):       Gemini 1.5 Flash
Production lokal (Traffic):  Llama 3.1 8B (vLLM)
Code-heavy:                  Claude 3.5 Sonnet
Multimodal:                  Gemini 1.5 Pro
Edge/Mobile:                 Phi-3 Mini
```

### Serving:
```
Development:     Ollama
Production GPU:  vLLM
Production CPU:  llama.cpp
Hugging Face:    TGI
```

### Quantization:
```
Ollama/CPU:     GGUF Q4_K_M / Q5_K_M
vLLM/GPU:       AWQ oder GPTQ
Fine-Tuning:    bitsandbytes 4-bit (QLoRA)
```

### Context:
```
RAG Standard:   4k-16k ausreichend
Lange Docs:     32k-128k (Llama, GPT-4, Claude)
Sehr lang:      1M (Gemini)
```

### Tools:
```
OpenAI API:     Native Function Calling ✅
Claude API:     Native Tools ✅
Llama 3.1:      Experimental Tools (Prompt Engineering besser)
Andere:         Prompt Engineering
```

---

**Navigation:**
- [← Back: Embedding Models](01-EMBEDDING-MODELS.md)
- [→ Next: Model Serving Deep-Dive](08-MODEL-SERVING.md) _(wenn du noch mehr zu Serving willst)_
- [← Back to Taxonomy](00-TAXONOMY.md)

**Version:** 1.0
**Last Updated:** 2025-10-03
**Maintainer:** ProduktRAG Project
