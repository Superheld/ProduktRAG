# Embeddings: Übersicht & Taxonomie

## Was sind Embeddings?

**Embeddings** sind numerische Repräsentationen von Text (oder anderen Daten) in einem hochdimensionalen Vektorraum. Sie transformieren menschenlesbare Informationen in eine Form, die Computer mathematisch verarbeiten können.

### Warum Embeddings?

**Problem:** Computer können nicht direkt mit Text arbeiten
```
"Laborkühlschrank mit 280 Liter Volumen"
→ Computer versteht das nicht
```

**Lösung:** Umwandlung in Zahlen
```
"Laborkühlschrank mit 280 Liter Volumen"
→ [0.23, -0.45, 0.12, 0.89, ..., -0.33]  # 768 Dimensionen
```

**Magie:** Ähnliche Texte haben ähnliche Vektoren
```
"Laborkühlschrank 280L"     → [0.23, -0.45, 0.12, ...]
"Medikamentenkühlschrank"   → [0.25, -0.43, 0.14, ...]  # Sehr ähnlich!
"Pommes Frites"             → [-0.89, 0.72, -0.34, ...] # Sehr unterschiedlich
```

---

## Embedding-Taxonomie

### 1. Dense Embeddings (Kontinuierlich)

**Charakteristik:**
- Alle Dimensionen haben Werte (keine Nullen)
- Werte meist zwischen -1 und 1
- Semantische Ähnlichkeit

**Beispiel:**
```python
dense_vector = [0.23, -0.45, 0.12, 0.89, -0.67, 0.34, ...]
# Jede Dimension hat einen Wert (768 oder 1024 Dimensionen typisch)
```

**Typische Models:**
- BERT, RoBERTa
- Sentence-Transformers (SBERT)
- E5, BGE
- MPNet

**Use-Cases:**
- ✅ Semantische Suche ("finde ähnliche Bedeutung")
- ✅ Frage-Antwort-Paare
- ✅ Paraphrasen-Erkennung
- ✅ Cross-Lingual Search

**Siehe:** [02-DENSE-EMBEDDINGS.md](02-DENSE-EMBEDDINGS.md)

---

### 2. Sparse Embeddings (Diskret)

**Charakteristik:**
- Meiste Dimensionen sind 0
- Nur wenige nicht-null Werte
- Keyword-basiert oder gelernt

**Beispiel:**
```python
sparse_vector = [0, 0, 3.2, 0, 0, 0, 1.5, 0, 0, ..., 0]
# Viele Nullen, nur relevante Terms haben Werte (oft 30k+ Dimensionen)
```

**Typen:**

**A) Classical Sparse (nicht gelernt):**
- **BM25** - Best Match 25 (Industry-Standard)
- **TF-IDF** - Term Frequency-Inverse Document Frequency

**B) Learned Sparse (neuronal):**
- **SPLADE** - Sparse Lexical and Dense Expansion
- Kombiniert Vorteile von Dense + Classical Sparse

**Use-Cases:**
- ✅ Exakte Keyword-Suche
- ✅ Technische Specs (Modellnummern, IDs)
- ✅ Named Entity Matching
- ✅ Lange Dokumente (>512 Tokens)

**Siehe:** [03-SPARSE-EMBEDDINGS.md](03-SPARSE-EMBEDDINGS.md)

---

### 3. Multi-Vector Embeddings (Token-Level)

**Charakteristik:**
- Nicht ein Vektor pro Dokument, sondern mehrere
- Jedes Token (Wort) wird einzeln embedded
- Interaktion auf Token-Ebene

**Beispiel:**
```python
# Standard Dense (1 Vektor):
doc = "Laborkühlschrank 280 Liter"
embedding = [0.23, -0.45, ...]  # 1 x 768

# Multi-Vector (viele Vektoren):
doc = "Laborkühlschrank 280 Liter"
embeddings = [
    [0.12, 0.34, ...],  # "Laborkühlschrank"
    [0.45, -0.23, ...], # "280"
    [-0.67, 0.89, ...]  # "Liter"
]  # 3 x 128
```

**Models:**
- **ColBERT** - Contextualized Late Interaction over BERT
- **Poly-Encoders**

**Use-Cases:**
- ✅ Präzise Passage-Retrieval
- ✅ Frage-Antwort mit exakter Stelle
- ✅ Wenn Dense nicht genau genug
- ⚠️ Mehr Speicher, langsamer

**Siehe:** [04-MULTI-VECTOR.md](04-MULTI-VECTOR.md)

---

### 4. Cross-Encoders (Joint Encoding)

**Charakteristik:**
- Kein echtes "Embedding" für Indexing
- Query + Document zusammen durchs Model
- Für Re-Ranking, nicht Retrieval

**Beispiel:**
```python
# Bi-Encoder (Standard):
embed_query = model.encode("Laborkühlschrank")
embed_doc = model.encode("Medizinisches Kühlgerät")
similarity = cosine(embed_query, embed_doc)

# Cross-Encoder:
score = model.encode_pair(
    "Laborkühlschrank",
    "Medizinisches Kühlgerät"
)  # Direkt Relevanz-Score
```

**Use-Cases:**
- ✅ Re-Ranking der Top-K Ergebnisse
- ✅ Höchste Genauigkeit nötig
- ❌ Zu langsam für große Datenmengen

**Siehe:** [05-CROSS-ENCODERS.md](05-CROSS-ENCODERS.md)

---

### 5. Hybrid Approaches (Kombination)

**Charakteristik:**
- Kombination von Dense + Sparse
- Nutzt Vorteile beider Welten
- Fusion-Strategien für Ranking

**Beispiel:**
```python
# Dense Score
dense_results = vector_db.search(dense_embedding, top_k=100)

# Sparse Score (BM25)
sparse_results = bm25.search(query_terms, top_k=100)

# Fusion (z.B. Reciprocal Rank Fusion)
final_results = rrf(dense_results, sparse_results)
```

**Fusion-Methoden:**
- **RRF** - Reciprocal Rank Fusion
- **Weighted Sum** - α * dense + (1-α) * sparse
- **Learned Fusion** - LLM entscheidet Gewichtung

**Use-Cases:**
- ✅ Beste Retrieval-Performance
- ✅ Robustheit gegen verschiedene Query-Typen
- ✅ Production-Systeme

**Siehe:** [06-HYBRID-APPROACHES.md](06-HYBRID-APPROACHES.md)

---

### 6. Quantized Embeddings (Komprimiert)

**Charakteristik:**
- Reduzierte Präzision für Speicher/Speed
- Von float32 → int8, binary, scalar
- Kaum Qualitätsverlust bei guter Quantisierung

**Typen:**

**A) Scalar Quantization (int8):**
```python
# Original (float32): 4 bytes pro Dimension
[0.234523, -0.456789, ...]

# Quantized (int8): 1 byte pro Dimension
[23, -46, ...]  # 4x weniger Speicher
```

**B) Binary Quantization:**
```python
# Original
[0.23, -0.45, 0.12, -0.67]

# Binary (nur +/-)
[1, 0, 1, 0]  # 32x weniger Speicher
```

**C) Matryoshka Embeddings (variable Dimensionen):**
```python
# Ein Model, mehrere Dimensionen
embedding_1024 = model.encode(text, dims=1024)
embedding_512 = embedding_1024[:512]  # Funktioniert trotzdem!
embedding_256 = embedding_1024[:256]
```

**Use-Cases:**
- ✅ Speicher-Constraints (Millionen Dokumente)
- ✅ Latenz-Optimierung
- ✅ Edge Devices (Handy, IoT)

**Siehe:** [07-QUANTIZATION.md](07-QUANTIZATION.md)

---

### 7. Vector Databases & Indexing

**Charakteristik:**
- Wie speichert man Millionen von Vektoren effizient?
- Index-Strukturen für schnelle Nearest-Neighbor-Suche
- Trade-off: Geschwindigkeit vs. Genauigkeit

**Index-Typen:**

**A) HNSW (Hierarchical Navigable Small World):**
- Graph-basiert
- Sehr schnell, hohe Recall
- Mehr Speicher

**B) IVF (Inverted File Index):**
- Clustering-basiert
- Weniger Speicher
- Langsamer als HNSW

**C) LSH (Locality Sensitive Hashing):**
- Hash-basiert
- Für sehr große Datenmengen
- Geringere Genauigkeit

**Beliebte Vector DBs:**
- **ChromaDB** - Einfach, lokal, ideal für Lernen
- **Qdrant** - Production-ready, Rust-basiert
- **Weaviate** - GraphQL, hybrid search
- **Pinecone** - Managed, cloud-only
- **Milvus** - Open-source, skalierbar

**Siehe:** [08-VECTOR-DATABASES.md](08-VECTOR-DATABASES.md)

---

## Entscheidungsbaum: Welches Embedding wann?

```
Frage 1: Was ist dein Use-Case?
│
├─ Semantische Suche, Ähnlichkeit
│  └─> Dense Embeddings (02-DENSE-EMBEDDINGS.md)
│
├─ Keyword-Matching, Exakte IDs/Specs
│  └─> Sparse Embeddings (03-SPARSE-EMBEDDINGS.md)
│
├─ Beide (Semantik + Keywords)
│  └─> Hybrid (06-HYBRID-APPROACHES.md)
│
└─ Höchste Präzision, Re-Ranking
   └─> Cross-Encoders (05-CROSS-ENCODERS.md)

Frage 2: Hast du Speicher/Latenz-Constraints?
│
└─ Ja
   └─> Quantization (07-QUANTIZATION.md)

Frage 3: Brauchst du Token-Level Matching?
│
└─ Ja
   └─> Multi-Vector (04-MULTI-VECTOR.md)
```

---

## Kapitel-Übersicht

| Kapitel | Thema | Wann lesen? |
|---------|-------|-------------|
| [01-FUNDAMENTALS.md](01-FUNDAMENTALS.md) | Vektorraum, Dimensionen, Distanzmetriken | **Zuerst!** Grundlagen verstehen |
| [02-DENSE-EMBEDDINGS.md](02-DENSE-EMBEDDINGS.md) | BERT, Sentence-Transformers | Standard für semantische Suche |
| [03-SPARSE-EMBEDDINGS.md](03-SPARSE-EMBEDDINGS.md) | BM25, TF-IDF, SPLADE | Für Keywords & technische Daten |
| [04-MULTI-VECTOR.md](04-MULTI-VECTOR.md) | ColBERT, Late Interaction | Advanced: Wenn Dense nicht reicht |
| [05-CROSS-ENCODERS.md](05-CROSS-ENCODERS.md) | Re-Ranking | Nach Retrieval für Top-K |
| [06-HYBRID-APPROACHES.md](06-HYBRID-APPROACHES.md) | Dense + Sparse Fusion | Production: Beste Performance |
| [07-QUANTIZATION.md](07-QUANTIZATION.md) | Kompression | Optimierung: Speicher/Speed |
| [08-VECTOR-DATABASES.md](08-VECTOR-DATABASES.md) | Indexing, Storage | Technische Implementation |

---

## Verwandte Kapitel (andere Ordner)

- **Training & Fine-Tuning:** Eigenes Kapitel (TODO)
- **Evaluation:** [_docs/evaluation/](../evaluation/00-overview.md)
- **Models:** [_docs/models/](../models/00-TAXONOMY.md)
- **Production:** [_docs/PRODUCTION-DEPLOYMENT.md](../PRODUCTION-DEPLOYMENT.md)

---

## Lern-Pfad Empfehlung

**Anfänger (neu bei Embeddings):**
1. 01-FUNDAMENTALS.md - Basics verstehen
2. 02-DENSE-EMBEDDINGS.md - Standard-Methode
3. 03-SPARSE-EMBEDDINGS.md - Alternative verstehen
4. 08-VECTOR-DATABASES.md - Wie speichern?

**Fortgeschritten (willst Production-System):**
1. 06-HYBRID-APPROACHES.md - Beste Retrieval-Qualität
2. 07-QUANTIZATION.md - Optimierung
3. 05-CROSS-ENCODERS.md - Re-Ranking

**Experte (Research/Cutting-Edge):**
1. 04-MULTI-VECTOR.md - Token-Level Matching
2. Alle anderen für vollständiges Bild

---

## Quick Reference: Embedding-Vergleich

| Typ | Dimensionen | Speicher | Speed | Use-Case |
|-----|-------------|----------|-------|----------|
| **Dense** | 384-1024 | Mittel | Schnell | Semantik |
| **Sparse** | 30k+ | Niedrig | Sehr schnell | Keywords |
| **Multi-Vector** | N x 128 | Hoch | Langsam | Präzision |
| **Cross-Encoder** | - | - | Sehr langsam | Re-Ranking |
| **Hybrid** | Dense + Sparse | Mittel | Mittel | Production |
| **Quantized** | 384-1024 (komprimiert) | Sehr niedrig | Sehr schnell | Scale |

---

*Nächster Schritt:* [01-FUNDAMENTALS.md](01-FUNDAMENTALS.md) - Verstehe die mathematischen Grundlagen
