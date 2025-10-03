# Embedding Models - Complete Deep Dive

**Zweck:** Alles was du über Embedding-Models für RAG wissen musst
**Scope:** Architekturen, Training, Auswahl, Evaluation, Fine-Tuning, Production
**Zielgruppe:** Entwickler die RAG-Systeme bauen und verstehen wollen wie Embeddings funktionieren

---

## 📋 Table of Contents

1. [Fundamentals](#1-fundamentals)
2. [Model Architectures](#2-model-architectures)
3. [Training Methods](#3-training-methods)
4. [Model Selection](#4-model-selection)
5. [Evaluation & Benchmarking](#5-evaluation--benchmarking)
6. [Fine-Tuning](#6-fine-tuning)
7. [Production Considerations](#7-production-considerations)
8. [Model Catalog](#8-model-catalog)
9. [Common Issues & Solutions](#9-common-issues--solutions)
10. [Advanced Topics](#10-advanced-topics)

---

## 1. Fundamentals

### 1.1 Was sind Embeddings?

**Definition:**
> Embeddings sind numerische Vektordarstellungen von Text, bei denen semantisch ähnliche Texte ähnliche Vektoren haben.

**Beispiel:**
```python
text1 = "Laborkühlschrank für Medikamente"
text2 = "Pharmazeutischer Kühlschrank"
text3 = "Automotor"

# Nach Embedding:
vec1 = [0.23, -0.45, 0.67, ...]  # 768 Dimensionen
vec2 = [0.24, -0.44, 0.68, ...]  # Ähnlich zu vec1
vec3 = [-0.89, 0.12, -0.34, ...] # Komplett anders

cosine_similarity(vec1, vec2) = 0.92  # Sehr ähnlich
cosine_similarity(vec1, vec3) = 0.15  # Nicht ähnlich
```

**Warum wichtig für RAG?**
1. **Semantic Search:** Finde ähnliche Texte ohne exakte Keywords
2. **Multilingualität:** "Refrigerator" findet "Kühlschrank"
3. **Synonyme:** "Medikament" findet "Pharmazeutikum"
4. **Kontext:** "Apple iPhone" ≠ "Apple Fruit"

---

### 1.2 Embedding Space

**Eigenschaften eines guten Embedding-Space:**

1. **Semantic Similarity**
   - Ähnliche Bedeutung → kleine Distanz
   - Unterschiedliche Bedeutung → große Distanz

2. **Clustering**
   - Konzepte gruppieren sich
   - "Kühlschrank", "Gefrierschrank", "Tiefkühler" nah beieinander

3. **Compositionality**
   - "König" - "Mann" + "Frau" ≈ "Königin"
   - In Praxis weniger stark als bei Word Embeddings

4. **Dimensionality**
   - Typisch: 384, 768, 1024, 1536, 3072 Dimensionen
   - Mehr Dimensionen = mehr Info, aber auch mehr Speicher/Rechenzeit

---

### 1.3 Distance Metrics

**Cosine Similarity** (Standard für Text-Embeddings)
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

# Range: -1 (gegensätzlich) bis +1 (identisch)
# Typisch für ähnliche Texte: 0.6 - 0.95
```

**Vorteile:**
- ✅ Unabhängig von Vektor-Länge
- ✅ Fokus auf Richtung, nicht Magnitude
- ✅ Standard in RAG-Systemen

**Euclidean Distance** (selten für Text)
```python
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# Kleinere Distanz = ähnlicher
```

**Vorteile:**
- ✅ Intuitiv geometrisch

**Nachteile:**
- ❌ Abhängig von Vektor-Länge
- ❌ Weniger robust für hochdimensionale Räume

**Dot Product** (schnell, aber nicht normalisiert)
```python
def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)
```

**Tipp:** Normalisiere Embeddings (L2-Normalization) → dann ist Dot Product = Cosine Similarity

```python
def normalize(vec):
    return vec / np.linalg.norm(vec)

# Dann: dot_product(normalize(v1), normalize(v2)) == cosine_similarity(v1, v2)
```

---

## 2. Model Architectures

### 2.1 BERT (Bidirectional Encoder Representations from Transformers)

**Architektur:**
```
Input: [CLS] Token1 Token2 ... TokenN [SEP]
  ↓
Transformer Encoder (12-24 Layers)
  ↓
Output: Vektor pro Token (768 dim für base, 1024 für large)
```

**Training:**
- **Masked Language Modeling (MLM):** 15% Tokens maskieren → vorhersagen
- **Next Sentence Prediction (NSP):** Zwei Sätze → zusammenhängend?

**Key Innovation:**
- **Bidirektional:** Sieht links UND rechts vom Token (GPT nur links→rechts)

**Für Embeddings:**
- ❌ **Problem:** Trainiert für Token-Prediction, nicht Sentence-Similarity
- ⚠️ **Lösung:** Pooling (Mean/CLS) um Sentence-Embedding zu bekommen
- ✅ **Besser:** Sentence-BERT nutzen (siehe unten)

**Beispiele:**
- `bert-base-uncased` (English, 110M Parameter, 768 dim)
- `dbmdz/bert-base-german-cased` (German, 110M Parameter)
- `deepset/gbert-large` (German, 335M Parameter, 1024 dim)

---

### 2.2 Sentence-BERT (SBERT)

**Architektur:**
```
Input Text
  ↓
BERT Encoder
  ↓
Pooling Layer (Mean/CLS/Max)
  ↓
Dense Layer (optional)
  ↓
Sentence Embedding (768 dim)
```

**Training:**
```
Anchor:   "Laborkühlschrank 280L"
Positive: "Kühlschrank für Labor mit 280 Liter"
Negative: "Bürostuhl ergonomisch"

→ Triplet Loss: minimize(distance(anchor, positive))
                maximize(distance(anchor, negative))
```

**Key Innovation:**
- **Siamese Network:** Zwei BERT-Models mit geteilten Gewichten
- **Contrastive Learning:** Lernt Similarity direkt
- **Pooling-Strategien standardisiert**

**Pooling-Strategien:**

1. **Mean Pooling** (am häufigsten)
```python
# Mittelwert aller Token-Embeddings
sentence_emb = mean(token_embeddings)
```
- ✅ Nutzt alle Tokens
- ✅ Robust

2. **CLS Pooling**
```python
# Nur [CLS] Token (Position 0)
sentence_emb = token_embeddings[0]
```
- ✅ Schnell
- ❌ Ignoriert andere Tokens

3. **Max Pooling**
```python
# Maximum pro Dimension
sentence_emb = max(token_embeddings, axis=0)
```
- ⚠️ Selten genutzt für Text

**Vorteile:**
- ✅ Optimiert für Sentence Similarity
- ✅ Schnelle Inferenz (separate Encoding)
- ✅ Einfache API (`sentence-transformers` Library)
- ✅ 1000+ vortrainierte Models

**Nachteile:**
- ❌ Bi-Encoder: Keine Cross-Attention zwischen Query & Doc
- ❌ Generisch trainiert (domain gap möglich)

**Beispiele:**
- `sentence-transformers/all-MiniLM-L6-v2` (English, 384 dim, sehr schnell)
- `sentence-transformers/all-mpnet-base-v2` (English, 768 dim, gute Quality)
- `paraphrase-multilingual-MiniLM-L12-v2` (50+ Sprachen, 384 dim)

---

### 2.3 E5 Models (Text Embeddings by Text-to-Text)

**Architektur:**
- Basis: T5 Encoder (Text-to-Text Transfer Transformer)
- Training: Contrastive Learning auf sehr großem Korpus

**Key Innovation:**
- **Prefix-based:** Query mit "query: " prefixen, Doc mit "passage: "
- **Multi-Task Training:** Verschiedene Retrieval-Tasks kombiniert
- **Sehr großer Trainingskorpus**

**Beispiele:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-large')

# WICHTIG: Prefix nutzen!
query = "query: Kühlschrank mit 280L"
docs = [
    "passage: Laborkühlschrank 280 Liter Volumen",
    "passage: Bürostuhl ergonomisch"
]

query_emb = model.encode(query)
doc_embs = model.encode(docs)
```

**Varianten:**
- `intfloat/e5-small-v2` (English, 384 dim, schnell)
- `intfloat/e5-base-v2` (English, 768 dim)
- `intfloat/e5-large-v2` (English, 1024 dim, beste Quality)
- `intfloat/multilingual-e5-large` (94 Sprachen, 1024 dim) ⭐ **Empfehlung**

**Vorteile:**
- ✅ State-of-the-art Performance (MTEB Benchmark)
- ✅ Multilingual (sehr gut für Deutsch)
- ✅ Großer Trainingskorpus → robust
- ✅ Prefix-Mechanismus → bessere Query-Doc-Unterscheidung

**Nachteile:**
- ❌ Prefix erforderlich (sonst schlechtere Performance)
- ❌ Größere Models langsamer als MiniLM

---

### 2.4 BGE Models (BAAI General Embedding)

**Architektur:**
- Basis: BERT / RoBERTa
- Training: Retromae Pre-training + Contrastive Fine-tuning

**Key Innovation:**
- **Retromae:** Encoder-Decoder Pre-training für bessere Representations
- **Hard Negatives:** Schwierige Negativ-Beispiele für robustes Training

**Varianten:**
- `BAAI/bge-small-en-v1.5` (English, 384 dim)
- `BAAI/bge-base-en-v1.5` (English, 768 dim)
- `BAAI/bge-large-en-v1.5` (English, 1024 dim)
- `BAAI/bge-m3` (Multilingual, 1024 dim)

**Vorteile:**
- ✅ Sehr gute Performance (Top 3 auf MTEB)
- ✅ Kein Prefix nötig (einfachere Nutzung als E5)
- ✅ Aktiv maintained (BAAI/Beijing Academy of AI)

**Nachteile:**
- ❌ Englisch-fokussiert (multilingual variant okay, aber nicht best-in-class)

---

### 2.5 OpenAI Embeddings (API)

**ada-002 (Legacy)**
```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Laborkühlschrank 280L"
)
embedding = response.data[0].embedding  # 1536 dim
```

**text-embedding-3-small / large (Aktuell)**
```python
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Laborkühlschrank 280L"
)
embedding = response.data[0].embedding  # 3072 dim
```

**Vorteile:**
- ✅ Sehr gute Quality (State-of-the-art)
- ✅ Kein lokales Setup
- ✅ Skaliert automatisch

**Nachteile:**
- ❌ Kostet Geld ($0.13/1M Tokens für large)
- ❌ Daten gehen zu OpenAI
- ❌ Latency durch API-Call
- ❌ Vendor Lock-in

**Kosten-Beispiel:**
```
10.000 Chunks à 200 Tokens = 2M Tokens
→ text-embedding-3-large: $0.26
→ text-embedding-3-small: $0.04
```

---

### 2.6 Cohere Embeddings (API)

**embed-multilingual-v3.0**
```python
import cohere
co = cohere.Client('YOUR_API_KEY')

response = co.embed(
    texts=["Laborkühlschrank 280L"],
    model="embed-multilingual-v3.0",
    input_type="search_document"  # oder "search_query"
)
embedding = response.embeddings[0]  # 1024 dim
```

**Key Features:**
- **Input Types:** `search_query`, `search_document`, `classification`, `clustering`
- **Compression:** Optional kleinere Dimensionen (128, 256, 512)

**Vorteile:**
- ✅ Sehr gut für Multilingual (100+ Sprachen)
- ✅ Input-Type-Optimierung
- ✅ Dimension-Compression ohne Re-Embedding

**Nachteile:**
- ❌ Kostenpflichtig
- ❌ API-abhängig

---

### 2.7 Specialized Models

#### **GerMedBERT** (Medizinisch, Deutsch)
```python
# Hypothetisch - prüfen ob verfügbar
model = SentenceTransformer('GerMedBERT/medbert-512')
```

**Use Case:**
- ✅ Deutsche medizinische Fachtexte
- ✅ Klinische Dokumentation
- ⚠️ Weniger Training-Daten als generische Models

#### **Legal-BERT**
- Juristische Texte
- Verträge, Urteile

#### **SciBERT**
- Wissenschaftliche Paper
- Technische Dokumentation

#### **CodeBERT**
- Source Code
- Code Search

**Wann Specialized Models?**
- ✅ Wenn Fachsprache stark von Alltagssprache abweicht
- ✅ Wenn Trainingsdaten in Domain verfügbar
- ❌ NICHT wenn generisches Model + Fine-Tuning besser

**Faustregel:**
> Starte mit generischem multilingual Model (E5-large) → Teste → Falls nötig: Fine-Tuning oder Specialized Model

---

## 3. Training Methods

### 3.1 Contrastive Learning

**Prinzip:**
```
Ähnliche Texte → kleine Distanz
Unähnliche Texte → große Distanz
```

**Triplet Loss:**
```python
anchor   = "Laborkühlschrank 280L"
positive = "Kühlschrank für Labor mit 280 Liter"
negative = "Bürostuhl ergonomisch"

# Loss Function:
loss = max(0, margin + distance(anchor, positive) - distance(anchor, negative))

# Margin = 0.5 typisch
# Ziel: distance(anchor, positive) < distance(anchor, negative) - margin
```

**Vorteile:**
- ✅ Direkt für Similarity optimiert
- ✅ Keine Labels nötig (nur Paare)

**Nachteile:**
- ❌ Hard Negatives schwer zu finden
- ❌ Viele Triplets nötig

---

### 3.2 In-Batch Negatives

**Prinzip:**
```
Batch = [
    (query1, doc1),  # Positive Pair
    (query2, doc2),  # Positive Pair
    (query3, doc3)   # Positive Pair
]

# Für query1:
# - doc1 ist Positive
# - doc2, doc3 sind Negatives (aus demselben Batch)

→ Effektiv: 1 Positive, (batch_size - 1) Negatives
```

**Vorteile:**
- ✅ Sehr effizient (keine separaten Negatives nötig)
- ✅ Große Batch Size → mehr Negatives → besseres Training

**Nachteile:**
- ❌ "Easy Negatives" (zufällig aus Batch)
- ⚠️ Braucht große Batches (32-512)

**Wird genutzt von:**
- E5 Models
- BGE Models
- Moderne Embedding-Models generell

---

### 3.3 Hard Negative Mining

**Prinzip:**
```
Query: "Laborkühlschrank 280L"

Easy Negative:  "Automotor V8"           ❌ zu einfach
Hard Negative:  "Gefrierschrank 300L"    ✅ ähnlich aber falsch
```

**Wie finden?**
1. **BM25 Retrieval:** Top-K mit Keyword-Search
2. **Falsche Treffer** = Hard Negatives

**Beispiel:**
```python
query = "Kühlschrank mit Temperaturüberwachung"

# BM25 findet:
# 1. Richtiger Kühlschrank mit Monitoring ✅ Positive
# 2. Anderer Kühlschrank OHNE Monitoring ✅ Hard Negative
# 3. Gefrierschrank mit Monitoring       ✅ Hard Negative
```

**Vorteile:**
- ✅ Model lernt feine Unterscheidungen
- ✅ Bessere Generalisierung

**Nachteile:**
- ❌ Aufwendig zu erstellen
- ❌ Braucht große Datenmengen

---

### 3.4 Knowledge Distillation

**Prinzip:**
```
Teacher Model (groß, gut, langsam)
    ↓ Embeddings als "Soft Labels"
Student Model (klein, schnell, weniger gut)

→ Student lernt Teacher zu imitieren
```

**Beispiel:**
```
Teacher: text-embedding-3-large (3072 dim, langsam)
Student: MiniLM-L6 (384 dim, schnell)

→ Student erreicht ~95% von Teacher Performance bei 5x Speed
```

**Anwendung:**
- **Compression:** Große Models → kleine Models
- **Multilingual Transfer:** English Model → German Model

**Vorteile:**
- ✅ Schnellere Models ohne viel Qualitätsverlust
- ✅ Weniger Trainingsdaten nötig

---

## 4. Model Selection

### 4.1 Decision Matrix

| Kriterium | Small (384 dim) | Medium (768 dim) | Large (1024+ dim) |
|-----------|----------------|------------------|-------------------|
| **Speed** | ⚡⚡⚡ Sehr schnell | ⚡⚡ Schnell | ⚡ Moderat |
| **Quality** | ⭐⭐ Gut | ⭐⭐⭐ Sehr gut | ⭐⭐⭐⭐ Exzellent |
| **Memory** | 💾 ~100MB | 💾 ~400MB | 💾 ~1GB |
| **Storage (Vectors)** | 💽 Klein | 💽 Mittel | 💽 Groß |

**Faustregel:**
```
< 10k Chunks:     Large (Quality > Speed)
10k - 100k:       Medium (Balance)
> 100k:           Small (Speed > Quality) oder Hybrid
Produktion:       Medium (sweet spot)
```

---

### 4.2 Konkrete Empfehlungen

#### **Für dein RAG-Projekt (2500 Chunks, Deutsch):**

**Option 1: `intfloat/multilingual-e5-large`** ⭐ **EMPFEHLUNG**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Mit Prefix!
query = "query: Kühlschrank mit 280L"
docs = ["passage: " + doc for doc in documents]

embeddings = model.encode(docs, show_progress_bar=True)
```

**Pro:**
- ✅ Top Performance auf MTEB (Multilingual)
- ✅ 1024 dim → gute Quality für technische Texte
- ✅ Gut für Deutsch
- ✅ 2500 Chunks → Speed kein Problem

**Contra:**
- ⚠️ Prefix erforderlich (nicht vergessen!)

---

**Option 2: `BAAI/bge-m3`**
```python
model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode(documents)  # Kein Prefix!
```

**Pro:**
- ✅ Sehr gute Performance
- ✅ Kein Prefix nötig
- ✅ Multilingual

**Contra:**
- ⚠️ Etwas schlechter für Deutsch als E5

---

**Option 3: `paraphrase-multilingual-mpnet-base-v2`**
```python
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(documents)
```

**Pro:**
- ✅ Gut getestet
- ✅ Schneller (768 dim)
- ✅ Solide Performance

**Contra:**
- ❌ Schlechter als E5/BGE

---

#### **Für Production (große Datenmengen):**

**`sentence-transformers/all-MiniLM-L6-v2`** (English only)
- 384 dim
- Sehr schnell
- Gut genug für viele Use Cases

**Hybrid Approach:**
- Kleine Model für Initial Retrieval (Top-100)
- Cross-Encoder für Re-Ranking (Top-10)

---

### 4.3 Multilingual Considerations

**Für deutsche + englische Texte:**
1. `intfloat/multilingual-e5-large` ⭐
2. `BAAI/bge-m3`
3. `paraphrase-multilingual-mpnet-base-v2`

**Für NUR deutsche Texte:**
- Überraschung: **Multilingual Models oft besser** als rein deutsche
- Grund: Mehr Trainingsdaten, bessere Generalisierung
- Ausnahme: Sehr spezifische Fachsprache (dann: Fine-Tuning)

---

## 5. Evaluation & Benchmarking

### 5.1 MTEB (Massive Text Embedding Benchmark)

**URL:** https://huggingface.co/spaces/mteb/leaderboard

**Tasks:**
1. **Retrieval** (RAG-relevant!)
2. Classification
3. Clustering
4. Re-Ranking
5. Semantic Textual Similarity (STS)
6. Summarization

**Top Models (Multilingual, Stand 2025):**
1. `intfloat/multilingual-e5-large` - 64.0 (Average)
2. `BAAI/bge-m3` - 63.5
3. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - 53.8

**Wichtig:**
- MTEB testet auf **englischen** Daten primär
- Multilingual Score ≠ Deutsch Score
- Für deutsche Performance: Eigene Tests nötig

---

### 5.2 Custom Evaluation für deine Domain

**Setup:**
```python
# 1. Test-Queries erstellen
test_queries = {
    'overview': "Welche Laborkühlschränke von Liebherr gibt es?",
    'technical': "Kühlschrank mit 280L und max. 150cm Höhe",
    'feature': "Wie funktioniert die Temperaturüberwachung?",
}

# 2. Ground Truth Labels (welche Docs sind relevant?)
ground_truth = {
    'overview': ['LABO-288', 'LABO-340', 'LABO-468'],  # Product IDs
    'technical': ['LABO-288', 'MED-COOL-300'],
    'feature': ['LABO-288', 'PHARMA-X500'],
}

# 3. Embeddings erstellen
query_embs = model.encode(list(test_queries.values()))
doc_embs = model.encode(all_documents)

# 4. Retrieval
from sklearn.metrics.pairwise import cosine_similarity

for query_idx, query_name in enumerate(test_queries.keys()):
    similarities = cosine_similarity(
        [query_embs[query_idx]],
        doc_embs
    )[0]

    # Top-5 Dokumente
    top_k = 5
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_ids = [doc_ids[i] for i in top_indices]

    # Metrics
    relevant = ground_truth[query_name]
    hits = len(set(retrieved_ids) & set(relevant))

    precision_at_k = hits / top_k
    recall_at_k = hits / len(relevant)

    print(f"{query_name}: P@{top_k}={precision_at_k:.2f}, R@{top_k}={recall_at_k:.2f}")
```

**Metrics:**

**Precision@K:**
```
P@5 = (Anzahl relevanter Docs in Top-5) / 5
```
- Wie viele der Ergebnisse sind relevant?
- Wichtig wenn User nur Top-K sieht

**Recall@K:**
```
R@5 = (Anzahl relevanter Docs in Top-5) / (Alle relevanten Docs)
```
- Wie viele relevante Docs wurden gefunden?
- Wichtig für vollständige Coverage

**MRR (Mean Reciprocal Rank):**
```
MRR = 1 / (Position des ersten relevanten Docs)

Beispiel:
Query 1: Erstes relevantes Doc auf Position 2 → 1/2 = 0.5
Query 2: Erstes relevantes Doc auf Position 1 → 1/1 = 1.0
MRR = (0.5 + 1.0) / 2 = 0.75
```

**NDCG (Normalized Discounted Cumulative Gain):**
- Berücksichtigt Ranking-Position
- Frühe Positionen wichtiger
- Standard für Retrieval-Evaluation

---

### 5.3 Similarity Distribution Analysis

**Nützlich um zu verstehen wie gut Embeddings clustern:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Alle Pairwise Similarities
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)

# Distribution plotten
plt.figure(figsize=(10, 6))
plt.hist(sim_matrix.flatten(), bins=100, alpha=0.7)
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Similarity Distribution')
plt.show()

# Erwartung:
# - Peak bei ~0.2-0.4 (unähnliche Docs)
# - Tail bei ~0.7-0.95 (ähnliche Docs)
# - Wenig bei 0.95+ (fast identische Docs = Duplikate?)
```

**Was sagt die Distribution?**

**Gute Distribution:**
```
|     ███
|    ████
|   █████         ██
| ███████       ████
|████████     ██████
+------------------------
0.0  0.3  0.6  0.9  1.0
```
- Klare Separation zwischen ähnlich/unähnlich
- Wenige Duplikate

**Schlechte Distribution:**
```
|        ████████████
|       █████████████
|      ██████████████
|    ████████████████
| ███████████████████
+------------------------
0.0  0.3  0.6  0.9  1.0
```
- Alles mittel-ähnlich
- Keine klare Unterscheidung
- Model nicht diskriminativ genug

---

## 6. Fine-Tuning

### 6.1 Wann Fine-Tuning?

**JA, wenn:**
- ✅ Domain stark unterschiedlich (Medizin, Recht, Technik)
- ✅ Spezielle Fachsprache / Jargon
- ✅ Genug Trainingsdaten (min. 1000 Paare)
- ✅ Generisches Model zeigt schlechte Performance

**NEIN, wenn:**
- ❌ Generisches Model funktioniert gut
- ❌ Wenig Trainingsdaten (<500 Paare)
- ❌ Keine Ressourcen für Training (GPU, Zeit)

**Faustregel:**
> Starte OHNE Fine-Tuning → Teste → Fine-Tune nur wenn nötig

---

### 6.2 Training Data Preparation

**Benötigt: Positive Paare**

**Quellen:**

**1. Existing Data (am besten)**
```python
# User Klicks = Implicit Feedback
pairs = [
    ("Kühlschrank mit Alarmfunktion", "LABO-288 PRO-ACTIVE Alarmüberwachung"),
    # Query → Clicked Document
]
```

**2. Synthetic Data (LLM-generiert)**
```python
# Nutze GPT/Claude um Queries zu generieren
doc = "Laborkühlschrank LABO-288 mit 280L Volumen und Temperaturüberwachung"

prompt = f"""
Generiere 5 realistische Suchqueries die zu diesem Dokument passen:
{doc}

Format: Eine Query pro Zeile.
"""

# Output:
# - Laborkühlschrank 280 Liter
# - Kühlschrank mit Temperaturüberwachung Labor
# - LABO-288 Spezifikationen
# ...
```

**3. Hard Negatives (wichtig!)**
```python
# Für jede Query: Ähnliche aber falsche Docs finden
query = "Kühlschrank mit Alarmfunktion"

positive_doc = "LABO-288 mit Alarmüberwachung"
hard_negative = "LABO-340 OHNE Alarm"  # Ähnlich aber fehlt Feature

# Triplet:
(query, positive_doc, hard_negative)
```

---

### 6.3 Fine-Tuning Methoden

#### **Methode 1: Sentence-Transformers Training**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Base Model laden
model = SentenceTransformer('intfloat/multilingual-e5-large')

# 2. Training Data
train_examples = [
    InputExample(texts=['query: Kühlschrank 280L', 'passage: LABO-288 280 Liter'], label=1.0),
    InputExample(texts=['query: Kühlschrank 280L', 'passage: Bürostuhl'], label=0.0),
    # ... viele mehr
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 3. Loss Function
train_loss = losses.CosineSimilarityLoss(model)

# 4. Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./fine-tuned-model'
)
```

**Loss Functions:**

**CosineSimilarityLoss:**
- Für Paare mit Similarity-Score (0.0 - 1.0)
- Einfach, effektiv

**MultipleNegativesRankingLoss:**
- Für Positive Paare (ohne explizite Negatives)
- Nutzt In-Batch Negatives
- **Empfohlen** für RAG

**TripletLoss:**
- Für Triplets (Anchor, Positive, Negative)
- Mehr Kontrolle über Negatives

---

#### **Methode 2: Adapter-basiertes Fine-Tuning (Parameter-efficient)**

```python
# Nur kleine Adapter-Layer trainieren statt ganzes Model
# → Weniger VRAM, schneller

# Beispiel mit PEFT Library
from peft import LoraConfig, get_peft_model

# ... (komplexer, für Advanced Use Cases)
```

**Vorteile:**
- ✅ Weniger VRAM (7B Model mit 6GB trainierbar)
- ✅ Schneller
- ✅ Mehrere Adapter für verschiedene Tasks

**Nachteile:**
- ❌ Etwas schlechtere Performance als Full Fine-Tuning
- ❌ Komplexer Setup

---

### 6.4 Training Best Practices

**Hyperparameter:**
```python
epochs = 3-5            # Mehr = Overfitting Risk
batch_size = 16-32      # Größer = besser (wenn VRAM erlaubt)
learning_rate = 2e-5    # Standard für BERT-based
warmup_steps = 10%      # von total steps
```

**Validation:**
```python
# Split Data:
# - 80% Training
# - 10% Validation (für Hyperparameter-Tuning)
# - 10% Test (finale Evaluation)

# Early Stopping:
# Stoppe wenn Validation Loss nicht mehr sinkt (3-5 Epochen)
```

**VRAM Requirements:**
```
Base Model Size → Full Fine-Tuning VRAM
- Small (110M):   ~6GB
- Base (340M):    ~12GB
- Large (560M):   ~20GB

Mit LoRA/Adapter: ~50% davon
```

---

### 6.5 Evaluation After Fine-Tuning

**Vergleiche:**
```python
# 1. Base Model Performance
base_model = SentenceTransformer('intfloat/multilingual-e5-large')
base_metrics = evaluate(base_model, test_queries, ground_truth)

# 2. Fine-Tuned Model Performance
ft_model = SentenceTransformer('./fine-tuned-model')
ft_metrics = evaluate(ft_model, test_queries, ground_truth)

# 3. Compare
print(f"Base Model - P@5: {base_metrics['p@5']:.3f}")
print(f"Fine-Tuned - P@5: {ft_metrics['p@5']:.3f}")
print(f"Improvement: {(ft_metrics['p@5'] - base_metrics['p@5']) / base_metrics['p@5'] * 100:.1f}%")
```

**Erwartete Improvements:**
- ✅ +5-15% auf Domain-spezifischen Queries
- ⚠️ Möglicherweise -2-5% auf generischen Queries (Trade-off)

**Red Flags:**
- ❌ Keine Verbesserung → zu wenig Daten oder schlechte Daten
- ❌ Overfitting → Validation Loss steigt, Training Loss sinkt
- ❌ Schlechter als Base → Bug im Training oder falsche Hyperparameter

---

## 7. Production Considerations

### 7.1 Inference Optimization

**Batch Processing:**
```python
# Schlecht: Ein Doc nach dem anderen
for doc in documents:
    emb = model.encode(doc)  # 2500x einzeln = langsam

# Gut: Batch Processing
batch_size = 32
embeddings = model.encode(
    documents,
    batch_size=batch_size,
    show_progress_bar=True
)
```

**Performance:**
```
Einzeln:  2500 docs × 20ms = 50 Sekunden
Batch32:  2500/32 × 150ms = 12 Sekunden
→ 4x schneller!
```

---

**GPU vs CPU:**
```python
import torch

# Check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

# Performance:
# GPU (RTX 3090): ~500 docs/sec
# CPU (i7-12700): ~50 docs/sec
# → 10x Unterschied
```

---

**Model Quantization (für CPU):**
```python
# ONNX Runtime (schneller auf CPU)
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained(
    'intfloat/multilingual-e5-large',
    export=True
)

# → ~2x schneller auf CPU
```

---

### 7.2 Caching Strategy

**Problem:**
> Embeddings neu zu berechnen ist teuer

**Lösung: Cache Embeddings**

**Offline Embedding (für Dokumente):**
```python
# Einmalig:
doc_embeddings = model.encode(documents)
np.save('embeddings.npy', doc_embeddings)

# Später:
doc_embeddings = np.load('embeddings.npy')
```

**Query Embedding (Runtime):**
```python
# Muss zur Laufzeit passieren (user input)
query_emb = model.encode(user_query)
```

**Optional: Query Cache (für häufige Queries):**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def embed_query(query: str):
    return model.encode(query)

# Häufige Queries werden gecached
```

---

### 7.3 Embedding Storage

**Formats:**

**NumPy (einfach):**
```python
import numpy as np

# Speichern
np.save('embeddings.npy', embeddings)  # ~10MB für 2500 × 1024

# Laden
embeddings = np.load('embeddings.npy')
```

**HDF5 (große Datenmengen):**
```python
import h5py

# Speichern
with h5py.File('embeddings.h5', 'w') as f:
    f.create_dataset('embeddings', data=embeddings)
    f.create_dataset('metadata', data=metadata)

# Laden (Memory-mapped, kein komplettes Laden nötig)
with h5py.File('embeddings.h5', 'r') as f:
    emb_subset = f['embeddings'][1000:2000]  # Nur Teil laden
```

**Pickle (mit Metadata):**
```python
import pickle

data = {
    'embeddings': embeddings,
    'doc_ids': doc_ids,
    'model_name': 'intfloat/multilingual-e5-large',
    'timestamp': '2025-10-03'
}

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)
```

---

### 7.4 Normalization

**Warum normalisieren?**
- Dot Product = Cosine Similarity (schneller!)
- Konsistente Ranges

**L2 Normalization:**
```python
import numpy as np

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Anwenden
embeddings_normalized = normalize(embeddings)

# Test:
vec = embeddings_normalized[0]
print(np.linalg.norm(vec))  # → 1.0 ✅
```

**In Sentence-Transformers:**
```python
# Automatisch normalisieren
embeddings = model.encode(texts, normalize_embeddings=True)

# Jetzt: dot(emb1, emb2) == cosine_similarity(emb1, emb2)
```

---

### 7.5 Monitoring & Versioning

**Model Metadata speichern:**
```json
{
  "model_name": "intfloat/multilingual-e5-large",
  "model_version": "v1.0",
  "model_hash": "sha256:abc123...",
  "embedding_dimensions": 1024,
  "num_documents": 2500,
  "normalization": "l2",
  "created_at": "2025-10-03T14:30:00Z",
  "git_commit": "a1b2c3d"
}
```

**Warum wichtig?**
- ✅ Reproduzierbarkeit
- ✅ Debugging (welches Model hat diese Embeddings erstellt?)
- ✅ A/B Testing (alte vs neue Embeddings vergleichen)

---

## 8. Model Catalog

### 8.1 Recommended Models (2025)

| Model | Dim | Languages | Speed | Quality | Use Case |
|-------|-----|-----------|-------|---------|----------|
| `intfloat/multilingual-e5-large` | 1024 | 94 | ⚡⚡ | ⭐⭐⭐⭐ | **RAG (multilingual)** ⭐ |
| `BAAI/bge-m3` | 1024 | 100+ | ⚡⚡ | ⭐⭐⭐⭐ | RAG (multilingual) |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | EN | ⚡⚡⚡ | ⭐⭐⭐ | Fast retrieval (EN) |
| `sentence-transformers/all-mpnet-base-v2` | 768 | EN | ⚡⚡ | ⭐⭐⭐⭐ | Balanced (EN) |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ | ⚡⚡ | ⭐⭐⭐ | Multilingual |
| `text-embedding-3-large` (OpenAI) | 3072 | 94 | ⚡ | ⭐⭐⭐⭐⭐ | Best quality (API, paid) |
| `embed-multilingual-v3.0` (Cohere) | 1024 | 100+ | ⚡⚡ | ⭐⭐⭐⭐ | Multilingual (API, paid) |

---

### 8.2 Specialized Models

**German-specific:**
- `deepset/gbert-large` (1024 dim, German BERT)
- `dbmdz/bert-base-german-cased` (768 dim)

**Medical:**
- `GerMedBERT/medbert-512` (German medical texts)
- `dmis-lab/biobert-v1.1` (English biomedical)

**Legal:**
- `nlpaueb/legal-bert-base-uncased`

**Code:**
- `microsoft/codebert-base`
- `Salesforce/codet5-base`

---

### 8.3 Model Comparison Tool

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_models(query, documents, models):
    """
    Compare different embedding models on same data
    """
    results = {}

    for model_name in models:
        print(f"Testing {model_name}...")
        model = SentenceTransformer(model_name)

        # Encode
        query_emb = model.encode(query)
        doc_embs = model.encode(documents)

        # Similarities
        sims = cosine_similarity([query_emb], doc_embs)[0]

        # Store
        results[model_name] = {
            'top_3_indices': np.argsort(sims)[::-1][:3],
            'top_3_scores': sorted(sims, reverse=True)[:3]
        }

    return results

# Usage:
models_to_test = [
    'intfloat/multilingual-e5-large',
    'BAAI/bge-m3',
    'paraphrase-multilingual-mpnet-base-v2'
]

query = "Kühlschrank mit Temperaturüberwachung"
documents = [
    "LABO-288 mit Alarmüberwachung und Temperaturmonitoring",
    "Bürostuhl ergonomisch verstellbar",
    "Gefrierschrank 300 Liter ohne Überwachung"
]

results = compare_models(query, documents, models_to_test)

for model, res in results.items():
    print(f"\n{model}:")
    for idx, score in zip(res['top_3_indices'], res['top_3_scores']):
        print(f"  [{score:.3f}] {documents[idx][:50]}")
```

---

## 9. Common Issues & Solutions

### 9.1 Poor Retrieval Quality

**Symptom:**
- Irrelevante Docs in Top-K
- Niedrige Precision/Recall

**Debugging:**

**1. Check Similarity Distribution**
```python
# Sind Similarities diskriminativ genug?
similarities = cosine_similarity(query_emb, doc_embs)
print(f"Min: {similarities.min():.3f}")
print(f"Max: {similarities.max():.3f}")
print(f"Mean: {similarities.mean():.3f}")
print(f"Std: {similarities.std():.3f}")

# Erwartung:
# Max ~ 0.7-0.95 (relevante Docs)
# Min ~ 0.1-0.3  (irrelevante Docs)
# Std ~ 0.1-0.2  (genug Variation)

# Schlecht:
# Max ~ 0.5, Min ~ 0.4, Std ~ 0.02  → keine Unterscheidung!
```

**2. Inspect Top Results**
```python
# Manuell die Top-10 anschauen
top_10 = np.argsort(similarities)[::-1][:10]
for idx in top_10:
    print(f"[{similarities[idx]:.3f}] {documents[idx][:100]}")

# Fragen:
# - Macht das Sinn?
# - Fehlt ein wichtiges Keyword?
# - Ist die Query zu vage?
```

**Lösungen:**

**A) Besseres Model**
```python
# Wechsel von MiniLM zu E5-large
model = SentenceTransformer('intfloat/multilingual-e5-large')
```

**B) Query-Expansion**
```python
# Original Query: "Kühlschrank 280L"
# Expanded: "Kühlschrank 280 Liter Volumen Kapazität"

# Mit LLM:
expanded_query = llm.generate(f"Erweitere diese Query mit Synonymen: {query}")
```

**C) Hybrid Search (Semantic + Keyword)**
```python
from rank_bff import fuse_ranks

# BM25 (Keyword)
bm25_results = bm25.get_top_k(query, k=20)

# Semantic
semantic_results = vector_db.query(query_emb, k=20)

# Fuse (RRF - Reciprocal Rank Fusion)
final_results = fuse_ranks([bm25_results, semantic_results])
```

**D) Fine-Tuning**
```python
# Wenn nichts hilft: Fine-Tune auf deiner Domain
```

---

### 9.2 Slow Embedding Generation

**Symptom:**
- 2500 Chunks → 5+ Minuten

**Solutions:**

**1. Batch Size erhöhen**
```python
# Von:
embeddings = model.encode(docs, batch_size=8)  # Langsam

# Zu:
embeddings = model.encode(docs, batch_size=64)  # Schneller

# VRAM-Check:
# 16GB VRAM → batch_size=128 möglich
# 8GB VRAM  → batch_size=32-64
# CPU       → batch_size=16-32
```

**2. GPU nutzen**
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('model-name', device=device)

# 10x speedup!
```

**3. Kleineres Model**
```python
# Von: e5-large (1024 dim, langsam)
# Zu:  e5-small (384 dim, 3x schneller)
# Trade-off: ~5% weniger Quality
```

**4. ONNX Runtime (für CPU)**
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
model = ORTModelForFeatureExtraction.from_pretrained(
    'intfloat/multilingual-e5-large',
    export=True
)
# ~2x schneller auf CPU
```

---

### 9.3 High Memory Usage

**Symptom:**
- Out of Memory (OOM) während Embedding
- Vector DB zu groß

**Solutions:**

**1. Streaming Embedding**
```python
# Nicht alles auf einmal:
chunks_per_batch = 500
all_embeddings = []

for i in range(0, len(docs), chunks_per_batch):
    batch = docs[i:i+chunks_per_batch]
    embs = model.encode(batch)
    all_embeddings.append(embs)

embeddings = np.vstack(all_embeddings)
```

**2. Float16 statt Float32**
```python
embeddings = model.encode(docs).astype(np.float16)

# Speicher: 50% weniger
# Quality-Loss: minimal (~0.1%)
```

**3. Dimensionality Reduction (PCA)**
```python
from sklearn.decomposition import PCA

# 1024 dim → 512 dim
pca = PCA(n_components=512)
embeddings_reduced = pca.fit_transform(embeddings)

# Speicher: 50% weniger
# Quality-Loss: ~3-5%
```

**4. Quantization (für Vector DB)**
```python
# In Qdrant/Weaviate: Scalar Quantization
# 768 float32 → 768 int8
# Speicher: 75% weniger
# Quality-Loss: 1-2%
```

---

### 9.4 Model Loading Issues

**Symptom:**
```
OSError: Can't load tokenizer for 'intfloat/multilingual-e5-large'
```

**Solutions:**

**1. Internet-Problem**
```bash
# Manuell herunterladen
huggingface-cli download intfloat/multilingual-e5-large

# Dann von Cache laden
model = SentenceTransformer('intfloat/multilingual-e5-large')
```

**2. Hugging Face Token (für private Models)**
```python
from huggingface_hub import login
login(token="hf_...")

model = SentenceTransformer('private/model-name')
```

**3. Offline Loading**
```python
# Model vorab herunterladen und speichern
model = SentenceTransformer('model-name')
model.save('/path/to/local')

# Später offline:
model = SentenceTransformer('/path/to/local')
```

---

## 10. Advanced Topics

### 10.1 Cross-Encoders (Re-Ranking)

**Bi-Encoder (Standard Embeddings):**
```
Query → Embedding → |
                     → Cosine Similarity
Doc   → Embedding → |

Vorteil: Schnell (separate Encoding)
Nachteil: Keine Interaktion zwischen Query & Doc
```

**Cross-Encoder (Re-Ranking):**
```
[Query, Doc] → BERT → Score (0-1)

Vorteil: Query & Doc interagieren (Cross-Attention) → bessere Accuracy
Nachteil: Langsam (N Docs = N Forward-Passes)
```

**Hybrid Pipeline:**
```python
from sentence_transformers import CrossEncoder

# 1. Bi-Encoder: Retrieve Top-100 (schnell)
bi_encoder = SentenceTransformer('intfloat/multilingual-e5-large')
query_emb = bi_encoder.encode(query)
top_100 = vector_db.query(query_emb, k=100)

# 2. Cross-Encoder: Re-Rank Top-100 → Top-10 (genau)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc] for doc in top_100]
scores = cross_encoder.predict(pairs)

# Top-10 nach Re-Ranking
top_10 = np.argsort(scores)[::-1][:10]
```

**Performance:**
- Bi-Encoder only: P@10 = 0.75
- Bi-Encoder + Cross-Encoder: P@10 = 0.85
- **+13% Improvement!**

**Trade-off:**
- Latency: +200-500ms (abhängig von Top-K)

---

### 10.2 Matryoshka Embeddings

**Konzept:**
> Ein Model das mehrere Dimensionen gleichzeitig unterstützt

**Beispiel:**
```python
# Model trainiert für 1024, 512, 256, 128, 64 dim

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

# Full Embeddings
full_emb = model.encode(text)  # 1024 dim

# Kleinere Embeddings (einfach truncate!)
emb_512 = full_emb[:512]   # 512 dim
emb_256 = full_emb[:256]   # 256 dim

# Quality bleibt gut!
```

**Vorteil:**
- ✅ Ein Model → mehrere Dimensionen
- ✅ Trade-off Speed vs Quality zur Laufzeit
- ✅ Kleinere Embeddings für Filtering, große für Final Ranking

**Use Case:**
```python
# Stage 1: Filter mit 256 dim (schnell)
candidates = fast_filter(emb_256, k=100)

# Stage 2: Re-Rank mit 1024 dim (genau)
final = re_rank(full_emb, candidates, k=10)
```

---

### 10.3 Multi-Vector Embeddings

**Standard (Single Vector):**
```
Doc = "Laborkühlschrank mit Alarmfunktion und Temperaturüberwachung"
  → ein Vektor [0.23, -0.45, ...]
```

**Multi-Vector (ColBERT):**
```
Doc = "Laborkühlschrank mit Alarmfunktion und Temperaturüberwachung"
  → ["Laborkühlschrank"]  → vec1
  → ["Alarmfunktion"]     → vec2
  → ["Temperatur"]        → vec3
  → ["überwachung"]       → vec4

→ Mehrere Vektoren pro Doc
```

**Matching:**
```python
# MaxSim: Maximum Similarity für jeden Query-Token
query_tokens = ["Kühlschrank", "Alarm"]
doc_tokens = [vec1, vec2, vec3, vec4]

score = 0
for q_tok in query_tokens:
    max_sim = max([cosine_sim(q_tok, d_tok) for d_tok in doc_tokens])
    score += max_sim

score /= len(query_tokens)
```

**Vorteil:**
- ✅ Granulare Matches (einzelne Konzepte)
- ✅ Bessere Accuracy

**Nachteil:**
- ❌ Viel mehr Storage (10x)
- ❌ Langsameres Retrieval

**Implementierungen:**
- ColBERT
- PLAID (schnellere Variante)

---

### 10.4 Late Interaction Models

**Konzept:**
- Query & Doc separat embedden (wie Bi-Encoder)
- Aber: Token-level Embeddings behalten
- Interaction zur Retrieval-Zeit (nicht Training-Zeit)

**Beispiel: ColBERTv2**
```python
# Offline: Embedde alle Docs (token-level)
doc_embeddings = model.encode_docs(documents)  # Shape: [N_docs, max_len, dim]

# Online: Query embedding
query_emb = model.encode_query(query)  # Shape: [query_len, dim]

# Late Interaction: MaxSim
scores = compute_maxsim(query_emb, doc_embeddings)
```

**Trade-off:**
- Quality: Zwischen Bi-Encoder und Cross-Encoder
- Speed: Schneller als Cross-Encoder, langsamer als Bi-Encoder
- Storage: Mehr als Bi-Encoder (token-level embeddings)

---

### 10.5 Embedding Compression Techniques

**1. Dimensionality Reduction (PCA)**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=512)
embeddings_compressed = pca.fit_transform(embeddings)

# 1024 → 512 dim: 50% Speicher, ~3% Quality-Loss
```

**2. Product Quantization (PQ)**
```python
# In Faiss
import faiss

# Original: 1024 float32 = 4096 bytes
# PQ: 1024 → 8 subvectors × 1 byte = 8 bytes
# → 500x Kompression!

index = faiss.IndexPQ(1024, 8, 8)  # dim, subvectors, bits
index.train(embeddings)
index.add(embeddings)
```

**3. Scalar Quantization (SQ)**
```python
# float32 → int8
# 4 bytes → 1 byte = 75% Speicher-Reduktion

embeddings_int8 = (embeddings * 127).astype(np.int8)

# Quality-Loss: 1-2%
```

**4. Binary Embeddings**
```python
# float → {-1, +1}
embeddings_binary = np.where(embeddings > 0, 1, -1)

# Speicher: 32x kleiner
# Quality-Loss: 10-20% (nur für sehr große Datenmengen sinnvoll)
```

---

## Summary: Quick Decision Guide

### Model Selection:
```
German RAG (2500 chunks):    intfloat/multilingual-e5-large
English RAG (large scale):   sentence-transformers/all-MiniLM-L6-v2
Multilingual RAG:            intfloat/multilingual-e5-large
Best Quality (cost ok):      text-embedding-3-large (OpenAI API)
Specialized Domain:          Fine-Tune multilingual-e5-large
```

### Fine-Tuning:
```
YES if: Domain-specific, >1000 pairs, base model <70% quality
NO if:  Base model works, <500 pairs, no GPU
```

### Production:
```
Batch Size:     32-64
Device:         GPU (10x faster)
Normalization:  L2 (enables dot product)
Storage:        NumPy (.npy) or HDF5 for large data
Caching:        Offline for docs, runtime for queries
```

### Optimization:
```
Speed:   Smaller model, ONNX, batching
Memory:  float16, PCA, streaming
Quality: Larger model, fine-tuning, cross-encoder re-ranking
```

---

**Navigation:**
- [← Back to Taxonomy](00-TAXONOMY.md)
- [→ Next: LLM Architectures](02-LLM-ARCHITECTURES.md)

**Version:** 1.0
**Last Updated:** 2025-10-03
**Maintainer:** ProduktRAG Project
