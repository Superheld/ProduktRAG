# Embedding Strategies for RAG Systems

Praxisorientiertes Nachschlagewerk für Embedding-Strategien in Retrieval-Augmented Generation Systemen.

---

## 📑 Inhaltsverzeichnis

1. [Embedding Fundamentals](#1-embedding-fundamentals) - Was sind Embeddings?
2. [Search Strategies](#2-search-strategies) - Wie suche ich?
3. [Query Optimization](#3-query-optimization) - Wie verbessere ich Queries?
4. [Evaluation](#4-evaluation) - Wie messe ich Qualität?
5. [Implementation Guide](#5-implementation-guide) - Praktische Umsetzung
6. [Decision Matrix](#6-decision-matrix) - Welche Strategie wann?

---

## 1. Embedding Fundamentals

**Worum geht's?** Grundlegende Konzepte und Typen von Embeddings verstehen.

### 1.1 Dense vs Sparse Embeddings

#### Dense Embeddings (Dichte Vektoren)

**Charakteristik:**
- Alle Dimensionen haben Werte (kontinuierlich)
- Meist normalisiert zwischen -1 und 1
- Typische Dimensionen: 384, 768, 1024

**Typische Models:**
- `sentence-transformers/all-mpnet-base-v2` (English, 768 dims, SOTA)
- `intfloat/multilingual-e5-large` (100+ Sprachen, 1024 dims)
- `deepset/gbert-large` (German, 1024 dims)
- `sentence-transformers/all-MiniLM-L6-v2` (Fast, 384 dims)

**Beispiel:**
```python
[0.234, -0.891, 0.456, 0.123, -0.567, 0.789, ...]
# 1024 Dimensionen, alle mit Werten
```

**Wie funktioniert's:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('deepset/gbert-large')
text = "Medikamentenkühlschrank für Impfstoffe"
embedding = model.encode(text, normalize_embeddings=True)

print(embedding.shape)  # (1024,)
print(embedding[:5])    # [0.234, -0.891, 0.456, 0.123, -0.567]
```

**Vorteile:**
- ✅ Semantisches Verständnis (erkennt Synonyme)
- ✅ Robust gegen Umformulierungen
- ✅ Findet konzeptuelle Ähnlichkeiten
- ✅ Sprachübergreifend möglich

**Nachteile:**
- ❌ Kann exakte Keyword-Matches "übersehen"
- ❌ Rechenintensiv (Matrix-Multiplikation für 1024 dims)
- ❌ Hoher Speicherbedarf (4KB pro Embedding bei float32)

**Wann nutzen:**
- Natürlichsprachliche Queries ("Welcher Kühlschrank für Impfstoffe?")
- Produktbeschreibungen, Reviews, Dokumentationen
- Konzeptuelle Suche statt exakte Matches

---

#### Sparse Embeddings (Dünnbesetzte Vektoren)

**Charakteristik:**
- Meist Nullen, wenige nicht-null Werte
- Hohe Dimensionalität (10k-100k)
- Werte repräsentieren Term-Wichtigkeit

**Typische Models/Methoden:**
- **BM25** (Statistisch, kein ML)
- **TF-IDF** (Klassisch, statistisch)
- **SPLADE** (`naver/splade-cocondenser-ensembledistil` - Learned Sparse)
- **DeepImpact** (Neural Sparse Retrieval)

**Beispiel:**
```python
[0, 0, 0, 3.2, 0, 0, 0, 0, 1.8, 0, 0, ...]
# 30k Dimensionen, nur ~20 non-zero
```

**Wie funktioniert's (BM25):**
```python
from rank_bm25 import BM25Okapi

corpus = [
    "Anzahl Schubfächer: 8",
    "Energieverbrauch: 172 kWh",
    "Temperaturbereich: +5 °C"
]

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "Schubfächer"
scores = bm25.get_scores(query.split())
# Output: [3.2, 0.0, 0.0] - nur erstes Dokument matched
```

**Vorteile:**
- ✅ Exakte Keyword-Matches
- ✅ Sehr schnell (nur non-zero Werte vergleichen)
- ✅ Interpretierbar (welches Wort matched?)
- ✅ Wenig Speicher (speichere nur non-zero)
- ✅ Keine ML nötig (BM25 ist statistisch)

**Nachteile:**
- ❌ Kein semantisches Verständnis
- ❌ Vocabulary Mismatch ("Kühlschrank" ≠ "Kühlgerät")
- ❌ Keine Synonyme
- ❌ Rechtschreibfehler sind fatal

**Wann nutzen:**
- Technische Spezifikationen mit exakten Begriffen
- Produktcodes, Normen (DIN 13277, HMFvh 4001)
- Wenn exakte Matches wichtiger als Semantik
- Supplement zu Dense Embeddings (Hybrid)

---

### 1.2 Embedding Granularity

**Auf welcher Ebene wird embedded?**

#### Word-Level Embeddings

**Konzept:** Jedes Wort einzeln als Vektor

**Typische Models:**
- **Word2Vec** (Google, 2013 - veraltet)
- **FastText** (Facebook, subword-aware)
- **GloVe** (Stanford, co-occurrence based)

**Beispiel:**
```python
# Word2Vec
"Kühlschrank" → [0.2, -0.5, 0.3, ...]
"Medikamente" → [0.1, -0.3, 0.8, ...]

# Sentence = Average of words
sentence_embedding = mean([word_emb1, word_emb2, word_emb3])
```

**Wann nutzen:**
- ❌ Nicht empfohlen für RAG (veraltet)
- Legacy-Systeme
- Sehr einfache Keyword-Suche

---

#### Sentence-Level Embeddings ⭐ **Empfohlen für RAG**

**Konzept:** Ganze Sätze/Chunks als ein Vektor

**Typische Models:**
- `sentence-transformers/all-mpnet-base-v2` (English SOTA)
- `intfloat/multilingual-e5-base` (100+ Sprachen)
- `deepset/gbert-large` (German)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (50+ Sprachen)

**Beispiel:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('deepset/gbert-large')

# Ganze Sentence wird als Einheit embedded
sentence = "Der Medikamentenkühlschrank hat 8 Schubfächer"
embedding = model.encode(sentence)
# Output: Single vector (1024,)
```

**Vorteile:**
- ✅ Kontext wird berücksichtigt (nicht nur einzelne Wörter)
- ✅ Standard für moderne RAG-Systeme
- ✅ Optimiert für Similarity-Search

**Wann nutzen:**
- ✅ **Default Choice für RAG**
- Chunks von 1-3 Sätzen (ideal: 100-300 tokens)
- Produktbeschreibungen, Specs

---

#### Document-Level Embeddings

**Konzept:** Ganzes Dokument als ein Vektor

**Typische Models:**
- `allenai/longformer-base-4096` (4096 Token Context)
- `allenai/led-base-16384` (16k Token Context)
- `google/long-t5-tglobal-base` (Long sequences)

**Beispiel:**
```python
# Gesamtes Dokument (mehrere Seiten)
document = "..." # 5000 words
embedding = long_model.encode(document)
```

**Wann nutzen:**
- Sehr lange Dokumente (Whitepapers, Manuals)
- Wenn Kontext über mehrere Seiten wichtig ist
- ⚠️ Meist nicht nötig für E-Commerce/Produktdaten

---

#### Token-Level Embeddings (Advanced)

**Konzept:** Sub-word Tokens (wie BERT intern arbeitet)

**Typische Models:**
- `colbert-ir/colbertv2.0` (Late Interaction)
- `answerdotai/answerai-colbert-small-v1` (Smaller variant)
- Custom BERT models mit Token-level output

**Beispiel:**
```python
# Jedes Token wird einzeln embedded
"Medikamentenkühlschrank"
→ ["Medikamente", "##n", "##kühl", "##schrank"]
→ [emb1, emb2, emb3, emb4]

# Interaction später (late interaction)
```

**Wann nutzen:**
- Sehr hohe Präzision nötig (Medical, Legal)
- Lange Dokumente mit spezifischen Passagen
- Advanced Use Case (komplexer)

---

### 1.3 Output Format & Pooling

**Wie kommt man von Tokens zu einem Sentence Embedding?**

#### Mean Pooling (Standard)

**Konzept:** Durchschnitt aller Token-Embeddings

```python
# BERT gibt Token-Embeddings aus
token_embeddings = model.encode_tokens("Kühlschrank für Impfstoffe")
# Shape: (5, 1024) - 5 tokens, je 1024 dims

# Mean Pooling
sentence_embedding = torch.mean(token_embeddings, dim=0)
# Shape: (1024,) - ein Vektor für ganze Sentence
```

**Vorteile:**
- ✅ Nutzt alle Tokens
- ✅ Robuster als nur CLS token
- ✅ Standard bei Sentence-Transformers

---

#### CLS Token Pooling

**Konzept:** Nur der [CLS] Token (erster Token bei BERT)

```python
# BERT hat speziellen [CLS] Token am Anfang
token_embeddings = model.encode_tokens("[CLS] Kühlschrank ...")
sentence_embedding = token_embeddings[0]  # Nur erster Token
```

**Wann nutzen:**
- BERT-Klassifikation (nicht Search)
- ⚠️ Nicht empfohlen für RAG

---

#### Max Pooling

**Konzept:** Maximum jeder Dimension über alle Tokens

```python
sentence_embedding = torch.max(token_embeddings, dim=0)
```

**Wann nutzen:**
- Selten in RAG
- Spezialisierte Use Cases

---

### 1.4 Normalization

**Sollen Embeddings normalisiert werden?**

**Ohne Normalisierung:**
```python
embedding = model.encode(text)
# Länge variiert: ||v|| = 3.2 oder 8.5
```

**Mit Normalisierung:**
```python
embedding = model.encode(text, normalize_embeddings=True)
# Länge = 1.0 (Unit vector)
# ||v|| = sqrt(v[0]² + v[1]² + ... + v[n]²) = 1.0
```

**Warum normalisieren?**
- ✅ Dot Product = Cosine Similarity (schneller)
- ✅ Alle Vektoren gleichberechtigt (nicht längenabhängig)
- ✅ Standard für Similarity Search

**Code:**
```python
model = SentenceTransformer('deepset/gbert-large')

# Immer normalisieren für RAG!
embeddings = model.encode(
    texts,
    normalize_embeddings=True  # ← Wichtig!
)
```

---

## 2. Search Strategies

**Worum geht's?** Verschiedene Ansätze um Embeddings für Retrieval zu nutzen.

### 2.1 Single Dense Search (Baseline)

**Konzept:** Ein Embedding-Model für alles

**Setup:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Model laden
model = SentenceTransformer('deepset/gbert-large')

# 2. Alle Chunks embedden
chunks = ["Chunk 1 text...", "Chunk 2 text...", ...]
embeddings = model.encode(chunks, normalize_embeddings=True)
np.save('embeddings.npy', embeddings)

# 3. Query embedden
query = "Medikamentenkühlschrank mit Alarm"
query_emb = model.encode(query, normalize_embeddings=True)

# 4. Similarity berechnen (Dot Product)
scores = embeddings @ query_emb

# 5. Top-K
top_k_indices = np.argsort(scores)[::-1][:5]
top_k_chunks = [chunks[i] for i in top_k_indices]
```

**Vorteile:**
- ✅ Einfach zu implementieren
- ✅ Gut für semantische Queries
- ✅ Ein Model = wenig Overhead

**Nachteile:**
- ❌ Exakte Keywords werden ggf. übersehen
- ❌ Ein Model muss alles können

**Wann nutzen:**
- Baseline/MVP
- Nur natürlichsprachliche Beschreibungen
- Keine technischen Specs mit exakten Begriffen

---

### 2.2 Hybrid Search (Dense + Sparse) ⭐

**Konzept:** Kombiniere semantische Suche (Dense) mit Keyword-Suche (BM25)

**Setup:**
```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# Dense Embeddings
model = SentenceTransformer('deepset/gbert-large')
dense_embeddings = model.encode(chunks, normalize_embeddings=True)

# Sparse Index (BM25)
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def hybrid_search(query, alpha=0.5, top_k=10):
    """
    Args:
        alpha: Weight for dense (1-alpha = sparse weight)
               alpha=1.0 → nur dense
               alpha=0.0 → nur sparse
               alpha=0.5 → 50/50
    """
    # Dense scores
    query_emb = model.encode(query, normalize_embeddings=True)
    dense_scores = dense_embeddings @ query_emb

    # Normalize dense scores to [0, 1]
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

    # Sparse scores (BM25)
    sparse_scores = bm25.get_scores(query.split())

    # Normalize sparse scores to [0, 1]
    sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())

    # Combine
    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    # Top-K
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return top_indices, hybrid_scores[top_indices]

# Usage
results, scores = hybrid_search(
    "Kühlschrank mit 8 Schubfächern",
    alpha=0.6  # 60% dense, 40% sparse
)
```

**Alpha-Tuning:**
```python
# Test verschiedene Alphas
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results = hybrid_search(query, alpha=alpha)
    precision = evaluate_precision(results)
    print(f"Alpha {alpha}: Precision = {precision:.2f}")

# → Finde optimales Alpha für deinen Use Case
```

**Wann nutzen:**
- ✅ **Empfohlen für technische Produktdaten**
- Mix aus Specs (exakte Begriffe) und Descriptions (semantisch)
- Produktcodes, Normen müssen exakt matchen
- Best of both worlds

**Vorteile:**
- ✅ Exakte Matches + Semantik
- ✅ Robust (wenn Dense versagt, hilft Sparse)
- ✅ Tunable (Alpha anpassbar)

**Nachteile:**
- ❌ Komplexer als Single Dense
- ❌ Zwei Systeme zu maintainen
- ❌ Alpha muss getunt werden

---

### 2.3 Multi-Model Strategy

**Konzept:** Verschiedene Models für verschiedene Datentypen

**Architecture:**
```python
MODELS = {
    "technical": SentenceTransformer("deepset/gbert-large"),
    "semantic": SentenceTransformer("intfloat/multilingual-e5-large"),
}

# Technische Specs
spec_chunks = ["Anzahl Schubfächer: 8", ...]
spec_embeddings = MODELS["technical"].encode(spec_chunks)

# Natürliche Beschreibungen
desc_chunks = ["Medikamentenkühlschrank mit SmartMonitoring...", ...]
desc_embeddings = MODELS["semantic"].encode(desc_chunks)

def multi_model_search(query, query_type="auto"):
    """
    Args:
        query_type: "technical" | "semantic" | "auto"
    """
    if query_type == "auto":
        # LLM klassifiziert Query (siehe Section 3.2)
        query_type = classify_query(query)

    if query_type == "technical":
        model = MODELS["technical"]
        embeddings = spec_embeddings
        chunks = spec_chunks
    else:
        model = MODELS["semantic"]
        embeddings = desc_embeddings
        chunks = desc_chunks

    query_emb = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_emb
    top_k = np.argsort(scores)[::-1][:10]

    return [chunks[i] for i in top_k]
```

**Wann nutzen:**
- Klar getrennte Datentypen (Specs vs Descriptions)
- Unterschiedliche Models performen besser auf verschiedenen Daten
- Genug Speicher für mehrere Models

**Vorteile:**
- ✅ Optimiert pro Datentyp
- ✅ Bessere Performance
- ✅ Flexibel erweiterbar

**Nachteile:**
- ❌ Mehr Speicher (mehrere Models geladen)
- ❌ Komplexer
- ❌ Query Classification nötig

---

### 2.4 SPLADE (Learned Sparse)

**Konzept:** Neural Network lernt sparse Vektoren mit semantischem Verständnis

**Special Feature:** Term Expansion
- Query: "Fächer"
- Expanded: ["Fächer", "Schubfächer", "Schubladen", "Ablageflächen"]
- Sparse wie BM25, aber mit ML-Power!

**Setup:**
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model_name = 'naver/splade-cocondenser-ensembledistil'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode_splade(text):
    """Erstellt sparse Vektor mit Term-Expansion"""
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)

    # Log-saturated activation
    vec = torch.max(
        torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1),
        dim=1
    )[0].squeeze()

    return vec  # Sparse vector (30k dims, nur ~20 non-zero)

# Alle Chunks encodieren
sparse_embeddings = [encode_splade(chunk) for chunk in chunks]

# Query
query_vec = encode_splade("Wie viele Fächer?")
# → matched "Fächer", "Schubfächer", "Ablageflächen"
```

**Wann nutzen:**
- Beste aus beiden Welten (Sparse + Semantic)
- Mehrsprachige technische Docs
- Wenn BM25 zu simpel, Dense zu langsam

**Vorteile:**
- ✅ Semantisches Verständnis + Sparse Efficiency
- ✅ Term Expansion automatisch
- ✅ Schneller als Dense (nur non-zero vergleichen)

**Nachteile:**
- ❌ Noch nicht so weit verbreitet
- ❌ Encoding langsamer als BM25
- ❌ Komplexere Implementierung

---

### 2.5 ColBERT (Late Interaction) - Advanced

**Konzept:** Query und Dokument werden auf Token-Level embedded, Interaction findet spät statt

**Unterschied zu Standard Dense:**
```python
# Standard Dense
query_embedding = model.encode("Wie viele Fächer?")  # → (1024,)
doc_embedding = model.encode("8 Schubfächer")        # → (1024,)
score = dot(query_embedding, doc_embedding)          # → Single score

# ColBERT
query_tokens = colbert.encode_query("Wie viele Fächer?")  # → (5, 128)
doc_tokens = colbert.encode_doc("8 Schubfächer")          # → (3, 128)
score = max_sim(query_tokens, doc_tokens)                 # → MaxSim über alle Token-Paare
```

**MaxSim Scoring:**
```python
def max_sim(query_tokens, doc_tokens):
    """
    Für jeden Query-Token: finde ähnlichsten Doc-Token
    """
    scores = []
    for q_token in query_tokens:
        # Similarity zu allen Doc-Tokens
        sims = [cosine_sim(q_token, d_token) for d_token in doc_tokens]
        scores.append(max(sims))  # Bester Match

    return sum(scores)  # Summe über alle Query-Tokens
```

**Wann nutzen:**
- Medical/Legal Domains (hohe Präzision nötig)
- Lange Dokumente (findet spezifische Passagen)
- Wenn Standard Dense zu grob ist

**Vorteile:**
- ✅ Feingranular (Token-Level statt Doc-Level)
- ✅ Bessere Precision
- ✅ Erklärbarer (welche Tokens matchen?)

**Nachteile:**
- ❌ Langsamer (mehr Tokens zu vergleichen)
- ❌ Mehr Speicher (alle Token-Embeddings)
- ❌ Komplexere Implementierung

---

### 2.6 Hierarchical Retrieval

**Konzept:** Zweistufige Suche: Erst grob (Produkte), dann fein (Details)

**Stage 1: Product-Level Retrieval**
```python
# Embed Product Summaries
product_summaries = [
    "HMFvh 4001: Medikamentenkühlschrank, 8 Fächer, 172 kWh",
    "HMFvh 5501: Medikamentenkühlschrank, 7 Fächer, 180 kWh",
    # ...
]
product_embeddings = model.encode(product_summaries)

# Find relevant products
query = "Kühlschrank mit etwa 8 Fächern"
query_emb = model.encode(query)
product_scores = product_embeddings @ query_emb
top_products = np.argsort(product_scores)[::-1][:5]
```

**Stage 2: Detail-Level Retrieval**
```python
# Suche nur in relevanten Produkten
detailed_results = []
for product_id in top_products:
    # Alle Chunks dieses Produkts
    product_chunks = get_chunks_for_product(product_id)
    chunk_embeddings = model.encode(product_chunks)

    # Detail-Suche
    scores = chunk_embeddings @ query_emb
    top_chunk_idx = np.argmax(scores)

    detailed_results.append({
        'product_id': product_id,
        'chunk': product_chunks[top_chunk_idx],
        'score': scores[top_chunk_idx]
    })

# Sortiere finale Ergebnisse
detailed_results.sort(key=lambda x: x['score'], reverse=True)
```

**Wann nutzen:**
- Große Datenmengen (10k+ Produkte)
- Performance wichtig (weniger Chunks zu durchsuchen)
- Klare Produkthierarchie

**Vorteile:**
- ✅ Schneller (weniger Vergleiche)
- ✅ Skaliert gut
- ✅ Kontext-aware (Details im Produkt-Kontext)

**Nachteile:**
- ❌ Komplexer
- ❌ Stage 1 Fehler propagieren (falsches Produkt → schlechte Details)

---

### 2.7 Multi-Vector Embeddings

**Konzept:** Ein Dokument → mehrere Embeddings (verschiedene Aspekte)

**Setup:**
```python
# Verschiedene Aspekte eines Produkts
product_data = {
    "title": "HMFvh 4001 H63 Perfection (+5° C)",
    "category": "Medikamentenkühlschrank",
    "key_specs": "8 Schubfächer, 172 kWh, DIN 13277",
    "description": "Medikamentenkühlschrank mit SmartMonitoring...",
    "safety_features": "Optische/akustische Alarme, Netzausfallalarm..."
}

# Embedde jeden Aspekt separat
embeddings = {
    "title": model.encode(product_data["title"]),
    "specs": model.encode(product_data["key_specs"]),
    "description": model.encode(product_data["description"]),
    "safety": model.encode(product_data["safety_features"])
}

# Bei Suche: gewichtete Kombination
def multi_vector_search(query, weights=None):
    if weights is None:
        weights = {"title": 0.3, "specs": 0.4, "description": 0.2, "safety": 0.1}

    query_emb = model.encode(query)

    score = sum(
        weights[aspect] * (embeddings[aspect] @ query_emb)
        for aspect in weights.keys()
    )

    return score

# Query-adaptive Weights
query = "Sicherheitsfeatures?"
weights = {"safety": 0.7, "specs": 0.2, "description": 0.1, "title": 0.0}
score = multi_vector_search(query, weights)
```

**Wann nutzen:**
- Komplexe Produkte mit vielen Aspekten
- Unterschiedliche Query-Intents (Safety vs Specs vs General)
- Wenn ein Embedding zu grob ist

**Vorteile:**
- ✅ Verschiedene Aspekte repräsentiert
- ✅ Flexibel gewichtbar
- ✅ Bessere Differenzierung

**Nachteile:**
- ❌ Mehr Speicher (mehrere Embeddings pro Doc)
- ❌ Komplexer
- ❌ Weights müssen getunt werden

---

## 3. Query Optimization

**Worum geht's?** Queries mit LLM verbessern bevor sie an Retrieval gehen.

### 3.1 Query Expansion

**Problem:**
```
User: "Wie viele Fächer?"
BM25: Findet "Anzahl Schubfächer: 8" nicht (Vocabulary Mismatch)
```

**Lösung: LLM erweitert Query**

**Code:**
```python
import anthropic

def expand_query_with_llm(user_query, available_spec_keys):
    """
    Erweitert Query mit domänenspezifischen Begriffen

    Args:
        user_query: Original User Query
        available_spec_keys: Liste von Spec-Keys aus Daten
                            ["Anzahl Schubfächer", "Energieverbrauch", ...]
    """
    client = anthropic.Anthropic(api_key="...")

    prompt = f"""
Du bist ein Assistent für technische Produktsuche.

User fragt: "{user_query}"

Verfügbare technische Eigenschaften in der Datenbank:
{json.dumps(available_spec_keys[:50], indent=2, ensure_ascii=False)}

Aufgabe: Generiere 3-5 Suchvarianten mit exakten Fachbegriffen.
Nutze die verfügbaren Eigenschaften.

Format: JSON-Array von Strings

Beispiel:
User: "Wie viele Fächer?"
Output: ["Anzahl Schubfächer", "Schubfächer Anzahl", "Fächer"]
"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    expanded_queries = json.loads(response.content[0].text)
    return expanded_queries

# Usage
available_keys = ["Anzahl Schubfächer", "Energieverbrauch", "Temperaturbereich", ...]
expanded = expand_query_with_llm("Wie viele Fächer?", available_keys)
# → ["Anzahl Schubfächer", "Schubfächer Anzahl", "Fächer", "Schubfach"]

# Suche mit allen Varianten
all_results = []
for variant in expanded:
    results = bm25.get_scores(variant.split())
    all_results.extend(results)

# Dedupliziere und ranke
final_results = deduplicate_and_rank(all_results)
```

**Vorteile:**
- ✅ Überbrückt Vocabulary Gap
- ✅ Nutzt Domain-Wissen aus LLM
- ✅ BM25 kann exakte Matches nutzen
- ✅ Funktioniert auch mit Dense (mehr Varianten)

**Nachteile:**
- ❌ LLM API Call nötig (Latenz + Kosten)
- ❌ Kann falsche Expansions generieren
- ❌ Query Drift Risiko (zu viele Begriffe)

---

### 3.2 Query Classification & Routing

**Konzept:** LLM entscheidet welche Suchstrategie optimal ist

**Code:**
```python
def classify_and_route_query(user_query):
    """
    Klassifiziert Query und routet zu optimaler Strategie
    """
    client = anthropic.Anthropic(api_key="...")

    prompt = f"""
Analysiere die Query: "{user_query}"

Kategorien:

A) TECHNICAL - Suche nach spezifischen technischen Daten
   Beispiele: "Wie viele Schubfächer?", "Energieverbrauch?", "Außenmaße?"
   → Nutze: BM25 + Query Expansion (exakte Begriffe wichtig)

B) SEMANTIC - Konzeptuelle/beschreibende Fragen
   Beispiele: "Welcher Kühlschrank für Impfstoffe?", "Sicherheitsfeatures?", "Was ist SmartMonitoring?"
   → Nutze: Dense Embeddings auf Beschreibungen

C) HYBRID - Kombination aus beidem
   Beispiele: "Energieeffizienter Kühlschrank mit Alarmsystem", "Kühlschrank mit 8 Fächern und WiFi"
   → Nutze: Hybrid Search (BM25 + Dense)

Antworte NUR mit: A, B oder C
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Schneller + günstiger
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    category = response.content[0].text.strip()

    # Route zu Strategie
    if category == "A":
        return search_technical(user_query)
    elif category == "B":
        return search_semantic(user_query)
    else:
        return search_hybrid(user_query)

def search_technical(query):
    """BM25 auf Specs mit Query Expansion"""
    expanded = expand_query_with_llm(query, available_spec_keys)
    results = bm25_search(expanded)
    return results

def search_semantic(query):
    """Dense auf Descriptions"""
    query_emb = model.encode(query)
    scores = description_embeddings @ query_emb
    return get_top_k(scores)

def search_hybrid(query):
    """Beide kombiniert"""
    dense_results = search_semantic(query)
    sparse_results = search_technical(query)
    return reciprocal_rank_fusion([dense_results, sparse_results])
```

**Vorteile:**
- ✅ Optimale Strategie pro Query
- ✅ Bessere Performance (nicht alles mit allem suchen)
- ✅ Ressourcen-effizient

**Nachteile:**
- ❌ LLM Call nötig (Latenz)
- ❌ Fehlklassifikation möglich
- ❌ Komplexer

---

### 3.3 Metadata-Guided Search

**Konzept:** LLM mapped User-Query auf strukturierte Metadaten

**Code:**
```python
# Deine Chunks haben Metadata
chunk_metadata_example = {
    "product_id": "HMFvh-4001-H63",
    "key": "Anzahl Schubfächer",
    "value": "8",
    "category": "specs"
}

# Alle verfügbaren Keys
available_keys = [
    "Anzahl Schubfächer",
    "Energieverbrauch in 365 Tagen",
    "Temperaturbereich",
    "Alarm bei Störung",
    "Schnittstelle",
    # ...
]

def metadata_guided_search(user_query, available_keys):
    """
    LLM mapped Query auf relevante Metadata-Keys
    """
    client = anthropic.Anthropic(api_key="...")

    prompt = f"""
User fragt: "{user_query}"

Verfügbare technische Eigenschaften:
{json.dumps(available_keys, indent=2, ensure_ascii=False)}

Welche Eigenschaften sind relevant für die User-Frage?

Output: JSON-Array der relevanten Keys (exakt wie oben)
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    relevant_keys = json.loads(response.content[0].text)

    # Direkte Filterung in Daten
    results = [
        chunk for chunk in all_chunks
        if chunk['metadata']['key'] in relevant_keys
    ]

    return results

# Example
results = metadata_guided_search(
    "Wie viele Fächer und welcher Energieverbrauch?",
    available_keys
)
# → Filtert Chunks mit keys: ["Anzahl Schubfächer", "Energieverbrauch in 365 Tagen"]
```

**Wann nutzen:**
- Strukturierte Daten mit klaren Keys
- Exakte Filterung wichtiger als Fuzzy Search
- Sehr schnell (kein Embedding-Vergleich nötig)

**Vorteile:**
- ✅ Sehr präzise
- ✅ Nutzt Struktur optimal
- ✅ Schnell (nur Metadata-Filter)
- ✅ Kein Embedding nötig

**Nachteile:**
- ❌ Nur für strukturierte Daten
- ❌ LLM Call nötig
- ❌ Funktioniert nicht für freie Descriptions

---

### 3.4 Reciprocal Rank Fusion (RRF)

**Konzept:** Kombiniere Ergebnisse von mehreren Retrieval-Strategien

**Problem:**
```
Dense Search:  [Doc5, Doc2, Doc8, Doc1, ...]
Sparse Search: [Doc2, Doc1, Doc5, Doc9, ...]

Welche sind die besten?
```

**Lösung: RRF**
```python
def reciprocal_rank_fusion(results_list, k=60):
    """
    Kombiniert mehrere Rankings zu einem

    Args:
        results_list: List of Lists [[doc_id, doc_id, ...], [...]]
        k: RRF parameter (standard: 60)

    Returns:
        Fused ranking
    """
    scores = {}

    for results in results_list:
        for rank, doc_id in enumerate(results, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0

            # RRF Formula
            scores[doc_id] += 1 / (k + rank)

    # Sort by score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in fused]

# Usage
dense_results = [5, 2, 8, 1, 3]
sparse_results = [2, 1, 5, 9, 7]

fused = reciprocal_rank_fusion([dense_results, sparse_results])
# → [2, 5, 1, 8, 9, 3, 7]
#    Doc 2 ist in beiden hoch ranked → top
```

**Warum RRF statt einfacher Kombination?**
```python
# Einfache Score-Addition (problematisch)
combined_scores = dense_scores + sparse_scores
# Problem: Unterschiedliche Skalen! (Dense: 0-1, BM25: 0-50)

# RRF: Unabhängig von Score-Skalen
# Nur Rank zählt!
```

**Wann nutzen:**
- Kombination mehrerer Retrieval-Methoden
- Unterschiedliche Score-Skalen
- Standard für Hybrid Search

---

## 4. Evaluation

**Worum geht's?** Wie messe ich ob meine Embedding-Strategie gut ist?

### 4.1 Semantic Similarity Tests

**Ziel:** Prüfen ob ähnliche Begriffe ähnliche Embeddings haben

**Code:**
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('deepset/gbert-large')

# Test-Paare
test_pairs = [
    # (term1, term2, expected_similarity)
    ("Medikamentenkühlschrank", "Pharma-Kühlgerät", "high"),
    ("Temperaturüberwachung", "SmartMonitoring", "medium"),
    ("Impfstoffe", "Vakzine", "high"),
    ("Anzahl Schubfächer", "8 Schubfächer", "high"),
    ("Energieverbrauch", "Preis", "low"),
    ("DIN 13277", "DIN13277", "high"),  # Format-Varianten
]

results = []
for term1, term2, expected in test_pairs:
    emb1 = model.encode(term1)
    emb2 = model.encode(term2)
    similarity = util.cos_sim(emb1, emb2).item()

    # Check if expectation met
    if expected == "high":
        passed = similarity > 0.7
    elif expected == "medium":
        passed = 0.4 < similarity < 0.7
    else:  # low
        passed = similarity < 0.4

    results.append({
        'pair': (term1, term2),
        'similarity': similarity,
        'expected': expected,
        'passed': passed
    })

    print(f"{term1:30s} <-> {term2:30s}: {similarity:.3f} ({'✓' if passed else '✗'})")

# Pass Rate
pass_rate = sum(r['passed'] for r in results) / len(results)
print(f"\nPass Rate: {pass_rate:.1%}")
```

**Target:** > 80% Pass Rate

---

### 4.2 Retrieval Evaluation

**Siehe [RAG-EVALUATION-GUIDE.md](RAG-EVALUATION-GUIDE.md) für Details**

**Quick Summary:**
```python
# 1. Ground Truth erstellen
ground_truth = {
    "query_1": {
        "query": "Kühlschrank mit 8 Fächern",
        "relevant_chunks": [42, 43, 155]  # IDs
    },
    # ...
}

# 2. Retrieval durchführen
results = retrieval_system.search("Kühlschrank mit 8 Fächern", top_k=10)

# 3. Metriken berechnen
precision_at_5 = len(set(results[:5]) & set([42, 43, 155])) / 5
recall_at_10 = len(set(results[:10]) & set([42, 43, 155])) / 3
mrr = 1 / (results.index(42) + 1)  # Position ersten relevanten

print(f"Precision@5: {precision_at_5:.2f}")
print(f"Recall@10: {recall_at_10:.2f}")
print(f"MRR: {mrr:.2f}")
```

**Key Metrics:**
- **Precision@K**: Von Top-K, wie viele relevant?
- **Recall@K**: Von allen relevanten, wie viele in Top-K?
- **MRR**: Position des ersten relevanten
- **NDCG**: Ranking-Qualität mit Relevanz-Graden

---

### 4.3 A/B Testing Strategies

**Code:**
```python
strategies = {
    "dense_only": lambda q: search_with_dense(q),
    "hybrid_50_50": lambda q: hybrid_search(q, alpha=0.5),
    "hybrid_70_30": lambda q: hybrid_search(q, alpha=0.7),
    "multi_model": lambda q: multi_model_search(q),
}

# Test Queries
test_queries = load_test_queries()  # Mit ground truth

results = {}
for name, search_fn in strategies.items():
    metrics = {
        'precision@5': [],
        'recall@10': [],
        'mrr': []
    }

    for test in test_queries:
        retrieved = search_fn(test['query'])
        relevant = test['relevant_chunks']

        metrics['precision@5'].append(precision_at_k(retrieved, relevant, 5))
        metrics['recall@10'].append(recall_at_k(retrieved, relevant, 10))
        metrics['mrr'].append(reciprocal_rank(retrieved, relevant))

    results[name] = {
        'precision@5': np.mean(metrics['precision@5']),
        'recall@10': np.mean(metrics['recall@10']),
        'mrr': np.mean(metrics['mrr'])
    }

# Visualisierung
import pandas as pd
df = pd.DataFrame(results).T
print(df)

#                 precision@5  recall@10   mrr
# dense_only          0.72       0.85    0.68
# hybrid_50_50        0.81       0.92    0.79
# hybrid_70_30        0.78       0.89    0.75
# multi_model         0.84       0.94    0.82  ← Winner!
```

---

## 5. Implementation Guide

**Worum geht's?** Schritt-für-Schritt Umsetzung für ProduktRAG.

### 5.1 Empfohlene Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                       User Query                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   LLM Query Classification         │
        │   (Technical / Semantic / Hybrid)  │
        └────────┬───────────────────────────┘
                 │
        ┌────────┴────────┬─────────────────┬──────────────┐
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐   ...
│  TECHNICAL    │  │  SEMANTIC    │  │   HYBRID     │
│               │  │              │  │              │
│ Query Expand  │  │  Dense E5    │  │  BM25 +      │
│      +        │  │      on      │  │  Dense GBERT │
│  BM25 Search  │  │ Descriptions │  │  Fusion      │
└───────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        └────────┬────────┴─────────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   Re-Ranking (LLM)     │
        │   Optional             │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   Top-K Results        │
        └────────────────────────┘
```

---

### 5.2 Step-by-Step Implementation

#### Phase 1: Baseline (Dense Only)

```python
# 1. Model laden
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('deepset/gbert-large')

# 2. Chunks laden
import json
chunks = []
with open('chunks_specs.jsonl') as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [chunk['document'] for chunk in chunks]

# 3. Embeddings generieren
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32
)

# 4. Speichern
import numpy as np
np.save('embeddings_gbert.npy', embeddings)

# 5. Search Function
def search(query, top_k=5):
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            'text': texts[i],
            'score': scores[i],
            'metadata': chunks[i]['metadata']
        }
        for i in top_indices
    ]

# Test
results = search("Medikamentenkühlschrank mit Alarm")
for r in results:
    print(f"{r['score']:.3f}: {r['text'][:100]}")
```

---

#### Phase 2: Hybrid Search

```python
from rank_bm25 import BM25Okapi

# 1. BM25 Index erstellen
tokenized = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized)

# 2. Hybrid Search
def hybrid_search(query, alpha=0.6, top_k=5):
    # Dense
    query_emb = model.encode(query, normalize_embeddings=True)
    dense_scores = embeddings @ query_emb
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

    # Sparse
    sparse_scores = bm25.get_scores(query.split())
    sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())

    # Combine
    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return [
        {
            'text': texts[i],
            'score': hybrid_scores[i],
            'dense_score': dense_scores[i],
            'sparse_score': sparse_scores[i],
            'metadata': chunks[i]['metadata']
        }
        for i in top_indices
    ]

# Test
results = hybrid_search("Kühlschrank mit 8 Schubfächern", alpha=0.6)
```

---

#### Phase 3: Query Classification

```python
import anthropic

client = anthropic.Anthropic(api_key="...")

def classify_query(query):
    prompt = f"""
Query: "{query}"

Kategorien:
A) TECHNICAL - Spezifische Daten (Anzahl, Maße, Verbrauch)
B) SEMANTIC - Konzeptuelle Fragen (Sicherheit, Anwendung)
C) HYBRID - Beides

Antwort (nur Buchstabe):
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()

def smart_search(query, top_k=5):
    category = classify_query(query)

    if category == "A":
        # Mehr Gewicht auf BM25
        return hybrid_search(query, alpha=0.3, top_k=top_k)
    elif category == "B":
        # Nur Dense
        return search(query, top_k=top_k)
    else:
        # Balanced Hybrid
        return hybrid_search(query, alpha=0.5, top_k=top_k)
```

---

### 5.3 Best Practices

**1. Normalisierung ist Pflicht**
```python
# ✅ Richtig
embeddings = model.encode(texts, normalize_embeddings=True)

# ❌ Falsch
embeddings = model.encode(texts)  # Längen variieren!
```

**2. Batch Processing**
```python
# ✅ Richtig (schnell)
embeddings = model.encode(texts, batch_size=32)

# ❌ Falsch (langsam)
embeddings = [model.encode(text) for text in texts]
```

**3. Caching**
```python
import os

if os.path.exists('embeddings.npy'):
    embeddings = np.load('embeddings.npy')
else:
    embeddings = model.encode(texts, normalize_embeddings=True)
    np.save('embeddings.npy', embeddings)
```

**4. Score Normalisierung für Hybrid**
```python
# Immer normalisieren vor Kombination!
dense_scores = (scores - scores.min()) / (scores.max() - scores.min())
```

**5. Top-K Tuning**
```python
# Zu wenig: Recall leidet
top_k = 3  # Nur für LLM mit sehr kleinem Context

# Zu viel: Precision leidet
top_k = 50  # LLM wird verwirrt

# Empfohlen:
top_k = 5  # Standard für RAG
```

---

## 6. Decision Matrix

**Worum geht's?** Welche Strategie für welchen Use Case?

### 6.1 Strategy Selection Table

| Use Case | Empfohlene Strategie | Alpha (Hybrid) | Begründung |
|----------|---------------------|----------------|------------|
| **Nur technische Specs** | Hybrid (BM25-heavy) | 0.3 | Exakte Begriffe wichtig (Produktcodes, Maße) |
| **Nur Beschreibungen** | Dense Only | N/A | Semantik wichtig, keine exakten Terms |
| **Gemischte Daten** | Hybrid (Balanced) | 0.5-0.6 | Best of both worlds |
| **Medical/Legal** | ColBERT + Reranking | N/A | Höchste Präzision nötig |
| **Große Datenmenge (10k+)** | Hierarchical | N/A | Performance & Skalierung |
| **Mehrsprachig** | E5 Multilingual + SPLADE | 0.5 | Cross-lingual semantics |
| **Budget-limitiert** | BM25 + GBERT | 0.4 | Keine LLM API Calls |
| **Höchste Präzision** | Multi-Model + LLM Routing | N/A | Verschiedene Strategien pro Query-Type |

---

### 6.2 Model Selection Guide

| Model | Dimensions | Language | Best For | Speed | Memory |
|-------|-----------|----------|----------|-------|--------|
| **deepset/gbert-large** | 1024 | DE | German technical docs | Medium | 1.3GB |
| **intfloat/multilingual-e5-large** | 1024 | 100+ | Cross-lingual, semantic | Medium | 1.4GB |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | EN | Fast baseline | Fast | 80MB |
| **GerMedBERT/medbert-512** | 768 | DE | Medical German | Medium | 500MB |
| **naver/splade-v2** | 30k sparse | Multi | Learned sparse | Slow | 400MB |

---

### 6.3 Für ProduktRAG (Medizintechnik)

**Deine Daten:**
- 4208 technische Specs (Key-Value Paare)
- 411 Produktbeschreibungen (natürlicher Text)
- Domain: Medizintechnik (Deutsch)
- Use Case: E-Commerce Produktsuche

**Empfohlene Strategie:**

```
✅ Hybrid Search (Dense + Sparse)
   - Dense: deepset/gbert-large für Descriptions
   - Sparse: BM25 für Specs
   - Alpha: 0.6 (60% Dense, 40% Sparse)

✅ LLM Query Classification
   - Route Technical Queries → BM25-heavy
   - Route Semantic Queries → Dense-only
   - Route Hybrid → Balanced

✅ Query Expansion (Optional)
   - Für technische Queries
   - Nutzt verfügbare Spec-Keys

✅ Metadata-Guided als Fallback
   - Wenn Query sehr spezifisch (z.B. "Produkt HMFvh 4001")
   - Direkte Filterung nach Product-ID
```

**Warum diese Wahl:**
1. **Medizintechnik** → Präzision wichtiger als Recall
2. **Mix aus Specs & Descriptions** → Hybrid nötig
3. **Deutsche Produkte** → GBERT optimal
4. **Exakte Begriffe in Specs** → BM25 unverzichtbar
5. **LLM verfügbar** → Query Optimization möglich

---

### 6.4 Implementation Roadmap

```
Phase 1: MVP (1 Woche)
├── ✅ Dense Embeddings (GBERT)
├── ✅ Basic Search
└── ✅ Evaluation Setup

Phase 2: Hybrid (1 Woche)
├── ⚡ BM25 Index
├── ⚡ Hybrid Search
├── ⚡ Alpha Tuning
└── ⚡ A/B Testing

Phase 3: LLM Integration (1 Woche)
├── ⚡ Query Classification
├── ⚡ Query Expansion
├── ⚡ Metadata Mapping
└── ⚡ Re-Ranking

Phase 4: Production (1 Woche)
├── ⚡ Vector DB (ChromaDB)
├── ⚡ API (FastAPI)
├── ⚡ Caching
└── ⚡ Monitoring

Phase 5: Advanced (Optional)
├── 💡 Multi-Model Strategy
├── 💡 ColBERT
├── 💡 Hierarchical Retrieval
└── 💡 SPLADE
```

---

## 📚 Resources

### Essential Libraries
```bash
pip install sentence-transformers
pip install rank-bm25
pip install chromadb
pip install anthropic
pip install numpy pandas scikit-learn
```

### Models (Hugging Face)
- **German**: `deepset/gbert-large`, `GerMedBERT/medbert-512`
- **Multilingual**: `intfloat/multilingual-e5-large`
- **Sparse**: `naver/splade-cocondenser-ensembledistil`
- **ColBERT**: `colbert-ir/colbertv2.0`

### Vector Databases
- **ChromaDB**: Einfach, lokal, perfekt für Prototyping
- **Qdrant**: Native Hybrid Search Support
- **Weaviate**: Multi-Vector Support
- **Pinecone**: Managed Service (Cloud)

### Weitere Docs
- [RAG-EVALUATION-GUIDE.md](RAG-EVALUATION-GUIDE.md) - Metriken & Evaluation
- [MODEL-ANALYSIS.md](MODEL-ANALYSIS.md) - Model-Vergleiche
- [ROADMAP.md](ROADMAP.md) - Projekt-Status

---

**Erstellt für:** ProduktRAG (Medizintechnik E-Commerce)
**Datum:** 2025-10-01
**Ziel:** Optimale Suche in technischen Spezifikationen und Produktbeschreibungen

*Last updated: 2025-10-01*
