# Model Analysis

Technischer Deep-Dive in Embedding-Modelle für RAG-Systeme: Architektur, Selection, Optimization.

---

## 📑 Inhaltsverzeichnis

1. [Model Architectures](#1-model-architectures) - Wie funktionieren verschiedene Modelle?
2. [Model Selection](#2-model-selection) - Welches Modell wählen?
3. [Benchmarking & Evaluation](#3-benchmarking--evaluation) - Wie vergleiche ich Modelle?
4. [Fine-Tuning](#4-fine-tuning) - Modelle anpassen
5. [Common Issues](#5-common-issues) - Typische Probleme & Lösungen
6. [Appendix: Model Catalog](#appendix-model-catalog) - Modell-Übersicht

**→ Production & Deployment:** Siehe [RAG-PRODUCTION-DEPLOYMENT.md](RAG-PRODUCTION-DEPLOYMENT.md)

---

## 1. Model Architectures

### BERT (Bidirectional Encoder Representations from Transformers)

**Architektur:**
- **Basis:** Transformer Encoder (12-24 Layer)
- **Attention:** Bidirektional - sieht gesamten Kontext links + rechts
- **Token:** [CLS] Token am Anfang, [SEP] zwischen Segmenten
- **Output:** Token-level Embeddings (ein Vektor pro Token)

**Training:**
- **Masked Language Modeling (MLM):** 15% der Tokens maskieren → Wort vorhersagen
- **Next Sentence Prediction (NSP):** Zwei Sätze → sind sie aufeinanderfolgend?
- **Trainingsdaten:** Wikipedia, BookCorpus (3.3B Wörter)

**Was macht BERT anders:**
- Erstes wirklich bidirektionales Modell (GPT nur links→rechts)
- Pre-Training + Fine-Tuning Paradigma etabliert
- Foundation für fast alle modernen NLP-Modelle

**Vorteile:**
- ✅ **Foundation Model:** Sehr robust, gut verstanden
- ✅ **Bidirektional:** Versteht Kontext aus beiden Richtungen
- ✅ **Viele Varianten:** Sprachen, Domains, Größen verfügbar
- ✅ **Stark für Classification:** Token/Sequence Classification, NER, QA

**Nachteile:**
- ❌ **Nicht für Similarity optimiert:** Training-Task = Wort-Vorhersage, nicht Satz-Ähnlichkeit
- ❌ **Token-Level Output:** Liefert Vektor pro Token, nicht pro Satz
- ❌ **NSP-Limitation:** Next Sentence Prediction weniger nützlich als gedacht

**Einsatzzwecke:**
- ✅ Named Entity Recognition (NER)
- ✅ Text Classification
- ✅ Question Answering
- ✅ Token-Level Tasks
- ⚠️ Sentence Embeddings (besser: Sentence-BERT nutzen)

**Beispiele:**
- `bert-base-uncased` (English, 110M Parameter)
- `dbmdz/bert-base-german-cased` (German, 110M Parameter)
- `deepset/gbert-large` (German, 335M Parameter)

---

### Sentence-BERT (SBERT)

**Architektur:**
- **Basis:** BERT Encoder + Pooling Layer
- **Siamese Network:** Zwei identische BERT-Modelle mit geteilten Gewichten
- **Output:** Sentence-level Embeddings (ein Vektor pro Satz)

**Training:**
- **Contrastive Learning:** Sentence-Triplets (Anchor, Positive, Negative)
- **Loss Function:** Triplet Loss oder Cosine Similarity Loss
- **Trainingsdaten:** NLI (Natural Language Inference), STS (Semantic Textual Similarity), QA-Pairs
- **Ziel:** Ähnliche Sätze → kleine Distanz, unähnliche → große Distanz

**Was macht SBERT anders:**
- BERT für Sentence Similarity optimiert (nicht nur Token-Prediction)
- Standardisiertes Framework für Embedding-Modelle
- Pooling-Strategien integriert (Mean/CLS/Max)

**Vorteile:**
- ✅ **Optimiert für Similarity:** Training mit Sentence-Paaren
- ✅ **Schnelle Inferenz:** Sentences einzeln embedden, dann vergleichen
- ✅ **Standardisiert:** Konsistente API für alle Modelle
- ✅ **Große Community:** HuggingFace Hub mit 1000+ Modellen
- ✅ **Einfache Nutzung:** `model.encode()` → fertig

**Nachteile:**
- ❌ **Kleinerer Trainingskorpus:** Als BERT Foundation Models
- ❌ **Domain-Gap:** Generisch trainiert (Wikipedia/NLI)
- ⚠️ **Bi-Encoder Limitation:** Query und Doc separat embedded (keine Cross-Attention)

**Einsatzzwecke:**
- ✅ Semantic Search & Retrieval (RAG!)
- ✅ Document Similarity & Clustering
- ✅ Duplicate Detection
- ✅ Information Retrieval
- ✅ Paraphrase Detection

**Beispiele:**
- `sentence-transformers/all-mpnet-base-v2` (English, 420M pairs)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (50+ Sprachen)
- `sentence-transformers/all-MiniLM-L6-v2` (Fast, 384 dims)

---

### RoBERTa (Robustly Optimized BERT)

**Architektur:**
- **Basis:** BERT Encoder (identisch)
- **Optimierungen:** Training-Prozess verbessert, nicht Architektur
- **Token:** Byte-Pair Encoding (BPE) statt WordPiece
- **Output:** Token-level Embeddings (wie BERT)

**Training:**
- **Nur MLM:** Kein NSP (Next Sentence Prediction)
- **Dynamic Masking:** Masken ändern sich bei jedem Epoch
- **Mehr Daten:** 160GB Text (vs BERT 16GB)
- **Größere Batches:** 8k Samples statt 256
- **Länger trainiert:** 500k steps statt 1M (aber effizienter)

**Was macht RoBERTa anders:**
- Entfernt NSP → NSP bringt keinen Mehrwert
- Dynamic Masking → bessere Generalisierung
- Beweis: Training-Hyperparameter wichtiger als Architektur

**Vorteile:**
- ✅ **Besser als BERT:** ~2-5% Performance-Gewinn bei gleicher Größe
- ✅ **Robuster:** Mehr Daten → stabilere Representations
- ✅ **Einfacher:** Kein NSP-Overhead
- ✅ **Flexibler:** Besseres Tokenization (BPE)

**Nachteile:**
- ❌ **Gleiche Limitation:** Token-level Output, nicht sentence-level
- ❌ **Nicht für Similarity optimiert:** Training = MLM
- ⚠️ **Weniger Domain-Varianten:** Vor allem English

**Einsatzzwecke:**
- ✅ Als Basis für Sentence-Transformers Fine-Tuning
- ✅ Text Classification (bessere Baseline als BERT)
- ✅ Token Classification / NER
- ✅ Sequence Labeling

**Beispiele:**
- `roberta-base` (English, 125M Parameter)
- `roberta-large` (English, 355M Parameter)
- `sentence-transformers/all-distilroberta-v1` (SBERT-Version, 82M Parameter)

---

### MPNet (Masked and Permuted Pre-training)

**Architektur:**
- **Basis:** Transformer Encoder
- **Hybrid:** Kombiniert BERT (MLM) + XLNet (Permutation Language Modeling)
- **Attention:** Bidirektional wie BERT
- **Output:** Token-level + Permutation-aware Representations

**Training:**
- **Masked & Permuted LM:** Wörter maskieren + Reihenfolge permutieren
- **Sieht kompletten Kontext:** Wie BERT, aber ohne [MASK] Artifakte
- **Dependency Modeling:** Bessere Position-Dependencies als BERT
- **Trainingsdaten:** Ähnlich wie RoBERTa (große Textkorpora)

**Was macht MPNet anders:**
- Kombiniert Vorteile von BERT (bidirektional) + XLNet (Permutation)
- Vermeidet [MASK] Token Bias
- Bessere Representation für lange Dependencies

**Vorteile:**
- ✅ **Beste BERT-Evolution:** Performance-Gewinn über BERT/RoBERTa
- ✅ **Keine [MASK] Artifakte:** Natürlichere Representations
- ✅ **SOTA (2021):** Bestes Balance Speed/Quality seiner Zeit
- ✅ **Gut für Sentence Embeddings:** Basis für `all-mpnet-base-v2`
- ✅ **Schnelle Inferenz:** Vergleichbar mit BERT/RoBERTa

**Nachteile:**
- ❌ **Weniger bekannt:** Kleinere Community als BERT
- ❌ **Weniger Domain-Varianten:** Hauptsächlich English General-Purpose
- ⚠️ **Überholt durch E5:** Neuere Modelle (2023+) teilweise besser

**Einsatzzwecke:**
- ✅ Sentence Embeddings (via Sentence-Transformers)
- ✅ Semantic Search (SOTA für viele Benchmarks)
- ✅ Text Classification
- ✅ Clustering & Similarity

**Beispiele:**
- `sentence-transformers/all-mpnet-base-v2` (English SOTA, 420M pairs)
- `microsoft/mpnet-base` (Base Model, 110M Parameter)

---

### E5 (Embeddings from bidirectional Encoder representations)

**Architektur:**
- **Basis:** Transformer Encoder (BERT-ähnlich)
- **Instruction-aware:** Prefix für Query/Passage Unterscheidung
- **Multi-Task Heads:** Verschiedene Task-spezifische Layer
- **Output:** Sentence-level Embeddings (768 dims base, 1024 dims large)

**Training:**
- **Multi-Task Learning:** Gleichzeitiges Training auf 10+ Tasks
- **Tasks:** QA, Paraphrase, NLI, Retrieval, Classification, etc.
- **1B+ Training Pairs:** Massive Datenmenge aus diversen Quellen
- **100+ Sprachen:** Multilingual von Anfang an
- **Contrastive Learning:** Mit Hard Negatives Mining

**Was macht E5 anders:**
- Massives Multi-Task Training (nicht nur ein Task)
- Instruction Prefixes: `query:` vs normale Passage
- Cross-lingual Transfer durch multilinguales Training

**Vorteile:**
- ✅ **Current SOTA:** Beste Performance auf vielen Benchmarks (2023+)
- ✅ **Massives Training:** 1B+ Pairs → sehr robuste Representations
- ✅ **Multilingual:** 100+ Sprachen in einem Modell
- ✅ **Instruction-aware:** Query vs Passage explizit unterschieden
- ✅ **Multi-Task:** Generalisiert hervorragend über verschiedene Tasks

**Nachteile:**
- ❌ **Größer:** Mehr Parameter → langsamere Inference
- ❌ **Instruction Prefix nötig:** `query:` Prefix für optimale Performance
- ❌ **Neuere Architektur:** Weniger Battle-tested als BERT/MPNet
- ⚠️ **Overkill für kleine Datasets:** Volle Power braucht größere Datenmengen

**Einsatzzwecke:**
- ✅ Multilingual Semantic Search
- ✅ Cross-lingual Retrieval
- ✅ Large-scale Production Systems
- ✅ Zero-shot Transfer zu neuen Sprachen
- ✅ Domain-agnostic Retrieval

**Beispiele:**
- `intfloat/multilingual-e5-large` (560M Parameter, 1024 dims)
- `intfloat/multilingual-e5-base` (278M Parameter, 768 dims)
- `intfloat/e5-large-v2` (English only, 335M Parameter)

---

### Model Architecture Comparison

| Model | Base Architecture | Training | Strengths | Weaknesses |
|-------|------------------|----------|-----------|------------|
| BERT | Transformer Encoder | MLM + NSP | Foundation, proven | Nicht similarity-optimiert |
| SBERT | BERT + Siamese | Contrastive | Similarity-optimiert | Kleiner Trainingskorpus |
| RoBERTa | BERT (improved) | MLM (optimized) | Bessere Performance | Größer |
| MPNet | BERT + XLNet | MLM + Permutation | SOTA (2021) | Weniger bekannt |
| E5 | Transformer | Multi-Task (massive) | SOTA (2023+), multilingual | Groß |

---

## 2. Model Selection

Die Wahl des richtigen Embedding-Modells ist entscheidend für die Performance deines RAG-Systems. Es gibt drei Hauptfaktoren: Domain, Resources, Performance.

---

### Auswahlkriterium 1: Domain Match

**Warum wichtig?**
Modelle lernen Vokabular und Semantik aus ihren Trainingsdaten. Domain-spezifische Modelle verstehen Fachterminologie, Abkürzungen und Kontext besser als General-Purpose Modelle.

**Training-Daten bestimmen Vokabular:**

| Typ | Trainingsdaten | Stärken | Schwächen |
|-----|---------------|---------|-----------|
| **General-Purpose** | Wikipedia, News, Web | Breites Allgemeinwissen | Flaches Fachvokabular |
| **Domain-Specific** | Medical, Legal, Technical Corpora | Tiefes Fachwissen, Synonyme | Schlechter auf anderen Domains |
| **Multilingual** | 100+ Sprachen | Cross-lingual Transfer | Weniger Tiefe pro Sprache |

**Entscheidungshierarchie:**

1. **Sprache:** Deutsch-only → German Models, Multilingual → E5/mE5, English → MPNet/E5
2. **Domain:** Medical → BioBERT/GerMedBERT, Legal → Legal-BERT, General → MPNet/E5
3. **Size/Speed:** Real-time → MiniLM, Offline → Large Models

**Typische Domain-Modelle:**
- **Medical:** BioBERT (EN), GerMedBERT (DE), PubMedBERT
- **Legal:** Legal-BERT (EN), German-Legal-BERT (DE)
- **Code:** CodeBERT, GraphCodeBERT
- **Scientific:** SciBERT, ScholarBERT
- **General:** MPNet (EN), GBERT (DE), E5 (Multi)

---

### Auswahlkriterium 2: Resource Constraints

**Warum wichtig?**
Größere Modelle = bessere Performance, aber auch mehr Memory und langsamere Inferenz. Production-Systeme haben harte Latency- und Memory-Limits.

**Size vs Performance Trade-off:**

| Model Size | Memory | Latency (CPU) | Performance | Einsatzszenario |
|------------|--------|---------------|-------------|-----------------|
| **Large** (>1GB) | 1-2GB | 150-250ms | Beste | Offline-Processing, Batch-Jobs |
| **Base** (~400MB) | 400-800MB | 40-80ms | Gut | Balanced Production APIs |
| **Small** (<100MB) | 80-200MB | 10-30ms | Akzeptabel | Real-time Search, Mobile, Edge |

**Memory-Komponenten:**

| Komponente | Berechnung | Beispiel (100k docs, 768 dims) |
|-----------|-----------|-------------------------------|
| Model Weights | Model-abhängig | ~400MB (base model) |
| Embeddings | N × dims × 4 bytes | 100k × 768 × 4 = ~300MB |
| Vector Index | +10-20% Overhead | ~30-60MB (FAISS HNSW) |
| **Total** | Summe | **~730-760MB** |

**Latency-Komponenten:**

1. **Embedding Time:** Modell-abhängig (10-250ms pro Query)
2. **Index Search:** Index-Typ abhängig (FAISS HNSW: <10ms, Flat: 50-200ms)
3. **Network/IO:** Datenbank-Zugriff, API-Calls
4. **Production Target:** Meist <300-500ms end-to-end

**Entscheidungshilfe:**
- **Real-time API (<100ms):** MiniLM, Distilled Models
- **Standard API (<300ms):** Base Models (MPNet, GBERT-Base)
- **High Quality (<500ms):** Large Models (E5-Large, GBERT-Large)
- **Batch Processing (no limit):** Largest Models für beste Qualität

---

### Auswahlkriterium 3: Performance Requirements

**Warum wichtig?**
Ein schnelles Modell nützt nichts bei schlechter Retrieval-Qualität. Die minimal akzeptable Performance bestimmt deine Modellwahl.

**Key Metriken:**

| Metrik | Was misst sie? | Wann wichtig? |
|--------|---------------|---------------|
| **Precision@K** | Anteil relevanter Docs in top-K | Precision-kritische Anwendungen (E-Commerce, User klickt top-3) |
| **Recall@K** | Anteil gefundener relevanter Docs | Recall-kritische Anwendungen (Medical, Legal - nichts verpassen!) |
| **MRR** | Position des ersten relevanten Docs | User Experience (schnell richtiges Ergebnis) |
| **nDCG@K** | Ranking-Qualität mit Relevance Scores | Ranking-Qualität wichtig |

**Typische Performance-Ranges:**

| Model Size | Precision@10 | Recall@10 | MRR | Trade-off |
|-----------|-------------|-----------|-----|-----------|
| **Large** | 0.80-0.90 | 0.70-0.80 | 0.75-0.85 | Beste Qualität, langsam |
| **Base** | 0.70-0.80 | 0.60-0.70 | 0.65-0.75 | Balanced |
| **Small** | 0.60-0.70 | 0.50-0.60 | 0.55-0.65 | Schnell, OK Qualität |

**Performance-Requirements setzen:**

1. **Baseline etablieren:** BM25 oder aktuelles System als Vergleich
2. **User Feedback:** Fehlertoleranz der Anwendung
3. **Business Impact:** Kosten von False Positives vs False Negatives

**Beispiele:**
- **Medical/Legal:** Hoher Recall (>0.7) → nichts verpassen ist kritisch
- **E-Commerce:** Hohe Precision (>0.8) → User klickt nur top-3 Results
- **Chatbot:** Hoher MRR (>0.7) → erste Antwort muss stimmen

---

### Domain Adaptation Strategies

Wenn kein passendes Domain-Modell existiert, oder es nicht gut genug performt, gibt es drei Strategien um ein generisches Modell anzupassen.

---

#### Strategie 1: Vocabulary Extension

**Wann nutzen?**
Domain-spezifische Begriffe werden als "Unknown Token" behandelt oder schlecht repräsentiert (z.B. Produktnummern, Normen, Fachbegriffe).

**Problem:**
```python
tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
tokens = tokenizer.tokenize("HMFvh-4001")
# → ['[UNK]', '-', '4001']  # "HMFvh" unbekannt!
```

**Lösung: Neue Tokens hinzufügen**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

# Wichtige Domain-Tokens identifizieren
new_tokens = ["HMFvh", "DIN58345", "Kühlmittel", "Blutkonserve"]
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens")

# Modell-Embedding-Layer anpassen
model = AutoModel.from_pretrained("deepset/gbert-base")
model.resize_token_embeddings(len(tokenizer))

# Jetzt: Training nötig um Embeddings für neue Tokens zu lernen
# (siehe Fine-Tuning Sektion 4)
```

**Vorteil:** Präzisere Repräsentation von Domain-Begriffen
**Nachteil:** Benötigt Re-Training

---

#### Strategie 2: Continued Pre-Training

**Wann nutzen?**
Du hast viele Domain-Texte (>10k Dokumente), aber keine labeled Paare für Fine-Tuning. Das Modell soll Domain-Sprachstil lernen.

**Was ist das?**
Weiteres Training des Basis-Modells (BERT) auf Domain-Daten mit Masked Language Modeling (MLM), bevor Sentence-Embedding Training.

**Warum sinnvoll?**
- Modell lernt Domain-Kontext: "Temperaturüberwachung" + "Alarm" kommen oft zusammen
- Verbessert interne Repräsentationen, bevor Similarity-Training
- Besonders wichtig wenn Domain-Vokabular stark abweicht

**Ablauf:**
1. **MLM auf Domain-Texten** (unsupervised)
2. **Sentence-Embedding Fine-Tuning** (mit Paaren, siehe Sektion 4)

```python
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. Modell laden
model = AutoModelForMaskedLM.from_pretrained("deepset/gbert-base")
tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

# 2. Domain-Texte vorbereiten
domain_texts = ["Text 1 aus Domain...", "Text 2...", ...]  # Viele!
tokenized = tokenizer(domain_texts, truncation=True, padding=True)

# 3. MLM Training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./gbert-medical-pretrained",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

trainer.train()

# 4. Danach: Sentence-Embedding Fine-Tuning (Sektion 4)
```

**Vorteil:** Starke Domain-Anpassung ohne Labels
**Nachteil:** Zeitaufwändig, braucht viel Daten

---

#### Strategie 3: Hybrid Multi-Model Ensemble

**Wann nutzen?**
Verschiedene Aspekte der Suche brauchen verschiedene Modelle. Beispiel: Allgemeine Semantik + Domain-Fachwissen.

**Konzept:**
Zwei Modelle parallel nutzen, Embeddings kombinieren.

**Varianten:**

**A) Concatenation** (Dimensionen addieren)
```python
# 1024 dim (general) + 512 dim (domain) = 1536 dim (combined)
combined = np.concatenate([general_emb, domain_emb], axis=1)
```
- ✅ Beide Aspekte erhalten
- ❌ Höhere Dimensionen → mehr Speicher

**B) Weighted Average** (Dimensionen bleiben)
```python
# Beide 1024 dim → 1024 dim (weighted)
combined = 0.6 * general_emb + 0.4 * domain_emb
```
- ✅ Kompakt
- ❌ Trade-off zwischen Aspekten

**Implementierung:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridEmbedder:
    def __init__(self, general_model_name, domain_model_name, weight_general=0.6):
        """
        Args:
            weight_general: 0.0-1.0, wie viel Gewicht auf general vs domain
        """
        self.general_model = SentenceTransformer(general_model_name)
        self.domain_model = SentenceTransformer(domain_model_name)
        self.weight_general = weight_general

    def encode(self, texts, normalize=True):
        # Beide Modelle embedden
        general_emb = self.general_model.encode(texts, normalize_embeddings=normalize)
        domain_emb = self.domain_model.encode(texts, normalize_embeddings=normalize)

        # Variante A: Concatenate
        # combined = np.concatenate([general_emb, domain_emb], axis=1)

        # Variante B: Weighted (Dimensionen müssen gleich sein!)
        combined = (self.weight_general * general_emb +
                   (1 - self.weight_general) * domain_emb)

        # Re-normalize nach weighted average
        if normalize:
            combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)

        return combined

# Nutzen
embedder = HybridEmbedder(
    general_model_name="deepset/gbert-large",
    domain_model_name="GerMedBERT/medbert-512",
    weight_general=0.6  # 60% general, 40% domain
)

embeddings = embedder.encode(["Medikamentenkühlschrank mit Alarm"])
```

**Gewichtung optimieren:**
```python
# A/B Test verschiedene Gewichte
for weight in [0.5, 0.6, 0.7, 0.8]:
    embedder.weight_general = weight
    p_at_10 = evaluate(embedder)  # Eigene Eval-Funktion
    print(f"Weight {weight}: P@10 = {p_at_10}")
```

**Vorteil:** Best of both worlds, flexibel
**Nachteil:** 2× Memory, 2× Inferenzzeit

---

## 3. Benchmarking & Evaluation

Wie vergleichst du objektiv verschiedene Modelle? Es gibt öffentliche Benchmarks (MTEB, BEIR) und eigene Evaluationen.

---

### Public Benchmarks: MTEB

**Was ist MTEB?**
Massive Text Embedding Benchmark - standardisierte Evaluation über 58 Datasets in 8 Task-Kategorien.

**Warum wichtig?**
- Vergleichbare Scores zwischen Modellen
- Verschiedene Aspekte getestet (nicht nur Retrieval)
- Community-Standard

**8 Task-Kategorien:**

| Kategorie | Datasets | Was wird getestet | Wichtig für RAG? |
|-----------|----------|-------------------|------------------|
| **Retrieval** | 15 | Dokumente zu Query finden | ✅✅✅ Sehr wichtig |
| Classification | 14 | Text-Kategorisierung | ⚠️ Weniger wichtig |
| Clustering | 11 | Ähnliche Texte gruppieren | ⚠️ Optional |
| Reranking | 4 | Top-K neu sortieren | ✅ Wichtig (wenn Re-Ranking) |
| STS | 10 | Semantic Textual Similarity | ✅ Wichtig (Ähnlichkeit) |
| Pair Classification | 3 | Duplicate Detection | ⚠️ Optional |
| Summarization | 1 | Summary vs Document | ❌ Nicht relevant |
| BitextMining | 4 | Translation Pairs | ❌ Nicht relevant |

**Für RAG fokussieren auf:**
- **Retrieval** (wichtigste!)
- STS (Semantic Similarity)
- Reranking (falls genutzt)

**Scores interpretieren:**
```python
# MTEB Leaderboard (Beispiel)
{
    "model": "intfloat/multilingual-e5-large",
    "avg_score": 64.41,     # ⚠️ Durchschnitt über ALLE 8 Kategorien
    "retrieval": 52.47,     # ✅ Das interessiert uns!
    "classification": 72.3, # ❌ Weniger relevant für RAG
    "clustering": 48.5,     # ❌ Weniger relevant für RAG
    "sts": 68.2            # ✅ Auch wichtig
}
```

**Achtung:** `avg_score` kann irreführend sein! Ein Modell kann gut in Classification aber schlecht in Retrieval sein. **Schau immer auf die Retrieval-Scores.**

**Selbst evaluieren:**
```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("deepset/gbert-large")

# Nur Retrieval-Tasks
evaluation = MTEB(tasks=["NFCorpus"])  # Biomedical Retrieval Dataset
results = evaluation.run(model, output_folder="results")

print(f"NDCG@10: {results['NFCorpus']['ndcg_at_10']}")
```

---

### Public Benchmarks: BEIR

**Was ist BEIR?**
Benchmarking Information Retrieval - spezialisiert auf Retrieval (direkter RAG-relevant).

**Unterschied zu MTEB:**
- MTEB: Breit (8 Kategorien)
- BEIR: Tief (nur Retrieval, aber 18 verschiedene Domains)

**Warum beide nutzen?**
- MTEB: Schneller Überblick (1-2 Tasks)
- BEIR: Detaillierte Retrieval-Analyse (mehrere Domains testen)

**BEIR Datasets (Auswahl):**

| Dataset | Domain | Dokumente | Queries | Use Case |
|---------|--------|-----------|---------|----------|
| MS MARCO | General Web | 8.8M | 6.9k | Generelle Websuche |
| Natural Questions | Wikipedia | 2.7M | 3.4k | Faktenwissen |
| HotpotQA | Multi-Hop | 5.2M | 7.4k | Complex Reasoning |
| FiQA | Finance | 57k | 648 | Financial QA |
| SciFact | Scientific | 5k | 300 | Science Papers |
| NFCorpus | Biomedical | 3.6k | 323 | Medical |

**Metriken:**
- **NDCG@10:** Normalized Discounted Cumulative Gain (berücksichtigt Ranking-Position)
- **MAP:** Mean Average Precision (Durchschnitt über alle Queries)
- **Recall@100:** Wie viele relevante Docs in top-100?

**NDCG@10 verstehen:**
```
Query: "Energieverbrauch Kühlschrank"
Relevante Docs: [5, 12, 23]

Ranking A: [5, 12, 23, ...]  → NDCG@10 = 1.0 (perfekt)
Ranking B: [1, 5, 12, ...]   → NDCG@10 = 0.85 (gut, relevante weiter oben)
Ranking C: [1, 2, 3, ..., 23] → NDCG@10 = 0.3 (schlecht, relevante weit unten)
```

**Wichtig:** NDCG belohnt relevante Docs weiter oben stärker (wie User-Verhalten).

---

### Eigene Domain Evaluation

**Warum eigene Evaluation?**
MTEB/BEIR sind generisch. Deine Domain ist spezifisch. **Beispiel:** BEIR hat kein Dataset für "deutsche Medizintechnik-Produktsuche".

**Vorgehen:**

**1. Ground Truth erstellen**
- 20-50 Test-Queries (verschiedene Typen)
- Für jede Query: Relevante Chunk-IDs annotieren
- Optional: Relevance Scores (0-3)

**2. Benchmark-Funktion schreiben**

```python
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

def benchmark_model(model_name, test_queries, corpus, ground_truth):
    """
    Evaluiert Modell auf eigenen Test-Daten.

    Args:
        model_name: Hugging Face Model ID
        test_queries: Liste von Query-Strings
        corpus: Liste von Dokument-Strings
        ground_truth: Liste von Listen (relevante Doc-IDs pro Query)

    Returns:
        Dict mit Precision@10, Recall@10, MRR
    """
    model = SentenceTransformer(model_name)

    # Corpus einmal embedden (Cache!)
    print(f"Embedding {len(corpus)} documents...")
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

    results = {"p@10": [], "r@10": [], "mrr": []}

    print(f"Evaluating {len(test_queries)} queries...")
    for query_id, query in enumerate(test_queries):
        # Query embedden
        query_emb = model.encode(query, convert_to_tensor=True)

        # Similarity berechnen
        scores = util.cos_sim(query_emb, corpus_embeddings)[0]

        # Top-10 retrieval
        top_k = torch.topk(scores, k=10)
        retrieved_ids = top_k.indices.cpu().tolist()

        # Ground Truth für diese Query
        relevant = ground_truth[query_id]

        # Precision@10: Von 10 Ergebnissen, wie viele relevant?
        relevant_in_top10 = len(set(retrieved_ids) & set(relevant))
        p_at_10 = relevant_in_top10 / 10
        results["p@10"].append(p_at_10)

        # Recall@10: Von allen relevanten, wie viele gefunden?
        r_at_10 = relevant_in_top10 / len(relevant) if len(relevant) > 0 else 0
        results["r@10"].append(r_at_10)

        # MRR: Position des ersten relevanten Ergebnisses
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                results["mrr"].append(1 / rank)
                break
        else:
            results["mrr"].append(0)  # Kein relevantes Ergebnis in top-10

    # Durchschnitt über alle Queries
    return {
        "precision@10": np.mean(results["p@10"]),
        "recall@10": np.mean(results["r@10"]),
        "mrr": np.mean(results["mrr"]),
        "num_queries": len(test_queries)
    }

# Beispiel Nutzung
test_queries = [
    "Energieverbrauch Kühlschrank",
    "Temperaturalarm Funktion"
]

corpus = [
    "Der Kühlschrank verbraucht 0.5 kWh pro Tag",  # 0
    "Alarm bei Temperaturabweichung",              # 1
    "Produktbeschreibung...",                       # 2
    # ... mehr Dokumente
]

ground_truth = [
    [0],      # Query 0: Nur Doc 0 relevant
    [1, 2]    # Query 1: Docs 1 und 2 relevant
]

# Evaluieren
scores = benchmark_model(
    model_name="deepset/gbert-large",
    test_queries=test_queries,
    corpus=corpus,
    ground_truth=ground_truth
)

print(f"Precision@10: {scores['precision@10']:.3f}")
print(f"Recall@10: {scores['recall@10']:.3f}")
print(f"MRR: {scores['mrr']:.3f}")
```

**Mehrere Modelle vergleichen:**
```python
models_to_test = [
    "deepset/gbert-large",
    "deepset/gbert-base",
    "intfloat/multilingual-e5-base"
]

comparison = {}
for model_name in models_to_test:
    print(f"\n=== Testing {model_name} ===")
    scores = benchmark_model(model_name, test_queries, corpus, ground_truth)
    comparison[model_name] = scores

# Bestes Modell finden
best_model = max(comparison.items(), key=lambda x: x[1]["mrr"])
print(f"\nBest model by MRR: {best_model[0]} ({best_model[1]['mrr']:.3f})")
```

---

### A/B Testing in Production

**Warum A/B Testing?**
Offline-Benchmarks (MTEB, eigene Eval) sind gut für initiale Auswahl, aber **echtes User-Verhalten** ist der ultimative Test. A/B Testing validiert Modell-Performance in Production.

**Grundkonzept:**

| Komponente | Beschreibung | Beispiel |
|-----------|--------------|----------|
| **Variant A** | Baseline (aktuelles Modell) | GBERT-Large |
| **Variant B** | Candidate (neues Modell) | E5-Large |
| **Traffic Split** | Prozent-Verteilung | 50/50 oder 90/10 (Conservative Rollout) |
| **Metrik** | Was misst Erfolg? | Click-Through-Rate, Clicked Rank, User Satisfaction |

**Wichtige Design-Prinzipien:**

1. **Consistent Hashing:** Gleicher User bekommt immer gleiche Variante
   - Warum: Vermeidet inkonsistente User Experience
   - Wie: Hash(user_id) % 100 bestimmt Variante

2. **Statistische Signifikanz:** Genug Samples für valide Aussage
   - Minimum: 1000+ Queries pro Variante
   - T-Test: p-value < 0.05 für 95% Konfidenz

3. **Metriken definieren:** Was ist "besser"?
   - **Click-Through-Rate (CTR):** % der Queries mit Klick
   - **Average Clicked Rank:** Position des geklickten Results (niedriger = besser)
   - **Session Success Rate:** % erfolgreicher Sessions

**Implementierung (Konzept):**

```python
# 1. Consistent Hashing für User-Zuweisung
variant = "a" if hash(user_id) % 100 < 50 else "b"  # 50/50 split

# 2. Modell-Auswahl basierend auf Variante
model = model_a if variant == "a" else model_b
embedding = model.encode(query)

# 3. Feedback loggen
log_click(user_id, variant, clicked_rank=3)  # User klickt Position 3

# 4. Nach N Queries: Auswertung
avg_rank_a = mean([1, 2, 3, 1, 2])  # Variante A: Ø 1.8
avg_rank_b = mean([2, 3, 4, 2, 3])  # Variante B: Ø 2.8
# → Variante A gewinnt (niedrigerer Rank = besser)
```

**Auswertung:**

| Schritt | Metrik | Formel | Interpretation |
|---------|--------|--------|----------------|
| **1. Deskriptiv** | Average Clicked Rank | mean(clicked_ranks) | Niedriger = besser |
| **2. Signifikanz** | T-Test | p-value < 0.05 | Ist Unterschied echt oder Zufall? |
| **3. Effektgröße** | Relative Improvement | abs(A - B) / A × 100 | Wie groß ist Verbesserung? |

**Entscheidungsmatrix:**

```
Results nach 1 Woche:
├─ Variante B besser UND p-value < 0.05
│  └─ ✅ Rollout zu 100% (B gewinnt)
├─ Variante B besser ABER p-value > 0.05
│  └─ ⚠️ Weiter testen (nicht signifikant)
└─ Variante A besser
   └─ ❌ Behalte A (B verliert
)
```

**Best Practices:**
- **Start Conservative:** 90/10 Split (90% Baseline, 10% Candidate) → bei Erfolg graduell erhöhen
- **Monitor Latency:** Neues Modell darf nicht signifikant langsamer sein
- **Rollback-Plan:** Bei negativen Metriken sofort zurück zu Baseline
- **Laufzeit:** Minimum 1 Woche für statistisch valide Aussage

---

## 4. Fine-Tuning

Fine-Tuning passt ein vortrainiertes Modell an deine spezifische Aufgabe an. Für Embeddings: Modell lernt, welche Texte ähnlich sind (für deine Domain).

---

### Wann lohnt sich Fine-Tuning?

**Decision Tree:**

```
Gibt es Domain-spezifisches Modell? (z.B. GerMedBERT für Medical)
├─ JA: Nutze das, kein Fine-Tuning nötig
└─ NEIN: Ist Performance-Gap zu groß?
    ├─ NEIN (<10% schlechter): Lohnt sich nicht
    └─ JA (>10% schlechter): Hast du >1000 Training-Paare?
        ├─ NEIN: Erst Daten sammeln oder Hybrid Search
        └─ JA: Fine-Tuning lohnt sich! →
```

**Kriterien im Detail:**

| Kriterium | Minimum | Optimal | Warum? |
|-----------|---------|---------|--------|
| Training-Paare | 1000+ | 10k+ | <500 → Overfitting-Risiko |
| Performance-Gap | >10% | >20% | Aufwand vs Nutzen |
| Domain-Vokabular | Viele UNK | Viele UNK | Modell kennt Begriffe nicht |
| Zeit/Ressourcen | GPU-Zugang | Multi-GPU | Training dauert Stunden-Tage |

**Beispiel-Entscheidung:**
```
Aktuell: Base Model, P@10 = 0.65
Ziel: P@10 > 0.75 (15% Verbesserung)
Daten: 3000+ annotierte Query-Doc-Paare vorhanden
Domain: Spezifische Domain mit Fachvokabular
→ ✅ Fine-Tuning sinnvoll!
```

---

### Fine-Tuning Methode 1: Contrastive Learning

**Konzept:**
Modell lernt durch Kontraste: Ähnliche Texte nah beieinander, unähnliche Texte weit auseinander im Embedding-Space.

**Training-Daten Formate:**

| Format | Struktur | Beispiel | Negatives |
|--------|----------|----------|-----------|
| **Paare** | (Query, Positive Doc) | ("search term", "relevant document") | Auto-generiert aus Batch |
| **Triplets** | (Anchor, Positive, Negative) | ("query", "relevant", "irrelevant") | Explizit angegeben |
| **Labeled Pairs** | (Text A, Text B, Ähnlichkeit) | ("doc1", "doc2", 0.8) | Label = Similarity Score |

**Loss-Funktionen:**

| Loss | Format | Vorteil | Nachteil | Empfehlung |
|------|--------|---------|----------|------------|
| **MultipleNegativesRankingLoss** | Paare | Einfach, gute Results | Negatives nur aus Batch | ⭐⭐⭐ Start hier |
| **TripletLoss** | Triplets | Kontrolle über Negatives | Mehr Daten-Annotation nötig | ⭐⭐ Für schwierige Cases |
| **ContrastiveLoss** | Labeled Pairs | Graduelle Ähnlichkeit | Braucht Similarity Scores | ⭐⭐ Für Ranking-Tasks |

**In-Batch Negatives:**
- Bei Paaren: Andere Docs im selben Batch = Negatives
- **Wichtig:** Batch Size erhöhen (32-64) für mehr diverse Negatives
- **Hard Negatives Mining:** Schwierige Negatives (ähnlich, aber falsch) gezielt hinzufügen für bessere Performance

**Workflow (vereinfacht):**

```python
# 1. Load base model
model = SentenceTransformer('base-model')

# 2. Prepare training data (Paare)
train_examples = [(query1, doc1), (query2, doc2), ...]  # 1000+ Paare

# 3. Define loss
loss = MultipleNegativesRankingLoss(model)

# 4. Train
model.fit(train_data, loss, epochs=3, batch_size=32)

# 5. Evaluate
precision = evaluate_on_test_set(model)
```

**Key Hyperparameter:**

| Parameter | Typische Werte | Einfluss | Tuning-Hinweis |
|-----------|---------------|----------|----------------|
| **Batch Size** | 16-64 | Mehr = mehr In-Batch Negatives | GPU Memory Limit beachten |
| **Learning Rate** | 1e-5 bis 5e-5 | Lerngeschwindigkeit | Zu hoch = instabil, zu niedrig = langsam |
| **Epochs** | 1-5 | Training-Dauer | >5 → Overfitting-Risiko |
| **Warmup Steps** | 10% der Steps | Stabilität am Anfang | Verhindert zu große Updates initial |

**Evaluation während Training:**
- Nach jedem Epoch auf Validation Set evaluieren
- Early Stopping: Training stoppen wenn Validation Score sich verschlechtert
- Best Model speichern (nicht letztes Epoch!)

---

### Fine-Tuning Methode 2: Triplet Loss

**Wann nutzen?**
Wenn **Hard Negatives** wichtig sind - Fälle wo ähnlich aussehende Texte unterschiedliche Bedeutung haben.

**Konzept:**
- **Anchor:** Query
- **Positive:** Relevantes Dokument
- **Negative:** Irrelevantes Dokument (idealerweise schwierig - ähnlich aber falsch)

**Datenformat:** (Anchor, Positive, Negative) Triplets

**Vorteil:** Explizite Kontrolle über was "unähnlich" ist
**Nachteil:** 3x mehr Daten-Annotation nötig als bei Paaren

**Hyperparameter:**
- **Triplet Margin:** Mindest-Abstand zwischen Positive und Negative (typical: 0.3-0.5)
- **Distance Metric:** Cosine oder Euclidean

---

### Fine-Tuning Methode 3: Hard Negatives Mining

**Problem:**
Random oder In-Batch Negatives sind oft zu einfach - Modell lernt nur grobe Unterschiede.

**Lösung:**
**Hard Negatives** = Dokumente die semantisch ähnlich aussehen, aber **nicht relevant** sind.

**Beispiel-Vergleich:**

| Typ | Beispiel | Schwierigkeit |
|-----|----------|---------------|
| **Easy Negative** | Query: "Product A specs" → Negative: "Company history" | ⭐ Offensichtlich unterschiedlich |
| **Hard Negative** | Query: "Product A specs" → Negative: "Product B specs" | ⭐⭐⭐ Ähnlich, aber falsch! |

**Mining-Prozess:**

1. **Embedde** alle Docs mit aktuellem Modell
2. **Sortiere** nach Similarity zur Query
3. **Filtere** relevante Docs raus
4. **Wähle** Top-K ähnlichste irrelevante Docs = Hard Negatives

**Workflow:**
```
Current Model → Embed Corpus → Find Similar-but-Irrelevant → Use as Hard Negatives → Fine-Tune
```

**Vorteil:** Modell lernt feine Unterscheidungen (z.B. Product A vs B)
**Aufwand:** Mining-Schritt vor Training + mehr Compute
**Performance-Gain:** ~5-15% über einfache In-Batch Negatives

---

### Fine-Tuning Methode 4: Distillation

**Wann nutzen?**
Großes Modell (Teacher) performt gut, aber ist zu langsam/groß für Production → komprimiere Wissen in kleines Modell (Student).

**Konzept:**
Student lernt Teacher-Embeddings zu imitieren (nicht Ground Truth Labels).

**Use Case Beispiel:**

| Szenario | Problem | Lösung via Distillation |
|----------|---------|------------------------|
| Production Latency | Large Model (560M params, 200ms) | Student (110M params, 50ms) mit ~90% Performance |
| Mobile/Edge | Model zu groß für Device | Distilled Model 3-5× kleiner |
| Cost Optimization | GPU-Inference zu teuer | Kleineres Model → CPU möglich |

**Workflow:**

```
1. Teacher (Large) → Embeddet Domain-Texte
2. Student (Small) → Lernt diese Embeddings zu reproduzieren (MSE Loss)
3. Result: Student mit ~85-95% Teacher Performance, 3-5× weniger Parameter
```

**Vorteile:**
- ✅ **Keine Labels nötig:** Nur unlabeled Domain-Texte (10k+)
- ✅ **Domain Adaptation:** Teacher-Wissen für deine spezifische Domain
- ✅ **Flexible Kompression:** Wähle Student-Größe nach Bedarf

**Nachteile:**
- ❌ **Braucht guten Teacher:** Distillation kann nicht besser werden als Teacher
- ❌ **Performance-Loss:** Typisch 5-15% schlechter als Teacher
- ⚠️ **Compute-Aufwand:** Teacher muss alle Texte embedden (einmalig, offline möglich)

---

## 5. Common Issues

Die häufigsten Probleme in der Praxis und wie du sie löst.

---

### Issue 1: Out of Memory (OOM)

**Wann tritt's auf?**
- Große Modelle (>1GB) auf schwacher GPU
- Große Batches beim Embedding/Training
- Zu viele Dokumente gleichzeitig im Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnose:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.1f} GB")
```

**Lösungen (nach Priorität):**

**1. Batch Size reduzieren:**
```python
# Statt:
embeddings = model.encode(texts, batch_size=32)

# Besser:
embeddings = model.encode(texts, batch_size=8)
```

**2. Gradient Accumulation (Fine-Tuning):**
```python
# Effektive Batch Size 32, aber nur 8 im Memory
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4  # 8 * 4 = 32
)
```

**3. Mixed Precision (FP16):**
```python
model.half()  # Float32 → Float16 (50% weniger Memory)
embeddings = model.encode(texts, convert_to_numpy=True)
```

**4. Smaller Model:**
```python
# Statt: gbert-large (1024 dim)
# Nutze: gbert-base (768 dim)
```

---

### Issue 2: Slow Inference

**Wann tritt's auf?**
- Production API mit <200ms Latency-Requirement
- Große Modelle (E5-Large, GBERT-Large)
- CPU statt GPU Inferenz
- Einzelne Texte statt Batches

**Symptom:**
```python
import time
start = time.time()
emb = model.encode("Test query")
print(f"Latency: {(time.time() - start) * 1000:.0f}ms")
# → 500ms (zu langsam für Production!)
```

**Lösungen (nach Impact):**

**1. Batch Processing:**
```python
# Langsam: Loop
embeddings = [model.encode(text) for text in texts]

# Schnell: Batch
embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
```

**2. GPU Utilization:**
```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('model-name', device=device)

# Multi-GPU
if torch.cuda.device_count() > 1:
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(texts, pool)
    model.stop_multi_process_pool(pool)
```

**3. ONNX Runtime:**
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Export to ONNX (schnellere Inferenz)
model = ORTModelForFeatureExtraction.from_pretrained(
    "deepset/gbert-base",
    export=True,
    provider="CUDAExecutionProvider"
)
```

**4. Quantization:**
```python
# INT8 Quantization (4x kleiner, schneller)
from sentence_transformers.quantization import quantize_embeddings

embeddings_float32 = model.encode(texts)
embeddings_int8 = quantize_embeddings(embeddings_float32, precision="int8")

# Search trotzdem möglich (leichter Performance-Loss)
```

---

### Issue 3: Poor Domain Performance

**Symptom:** Generisches Modell performt schlecht auf Domain-Daten

**Debug:**
```python
# Test Similarity auf Domain-Begriffen
from sentence_transformers import util

model = SentenceTransformer('model-name')

pairs = [
    ("Medikamentenkühlschrank", "Kühlschrank für Medikamente"),  # Sollte hoch sein
    ("DIN 58345", "DIN58345"),  # Sollte hoch sein (gleich)
    ("HMFvh 4001", "HMFvh 4011"),  # Sollte mittel sein (ähnlich aber nicht gleich)
]

for text1, text2 in pairs:
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    sim = util.cos_sim(emb1, emb2).item()
    print(f"{text1} <-> {text2}: {sim:.3f}")
```

**Lösungen:**

**1. Domain-Modell nutzen:**
```python
# Statt: deepset/gbert-base
# Nutze: GerMedBERT/medbert-512 (wenn Medical)
```

**2. Fine-Tuning (siehe Sektion 4)**

**3. Hybrid Search:**
```python
# Dense für Semantik + Sparse für exakte Matches
from rank_bm25 import BM25Okapi

# Dense
dense_scores = embeddings @ query_emb

# Sparse (BM25)
bm25 = BM25Okapi([doc.split() for doc in corpus])
sparse_scores = bm25.get_scores(query.split())

# Combine
final_scores = 0.7 * dense_scores + 0.3 * sparse_scores
```

---

### Issue 4: High Similarity Scores (Undifferenziert)

**Symptom:** Alle Scores >0.9, schwer zu ranken

**Ursache:** Modell zu generisch oder Texte zu ähnlich

**Lösungen:**

**1. Normalisierung prüfen:**
```python
# Embeddings sollten normalisiert sein
norms = np.linalg.norm(embeddings, axis=1)
print(f"Min norm: {norms.min()}, Max norm: {norms.max()}")
# Sollte: ~1.0 wenn normalisiert

# Falls nicht:
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

**2. Länge-Normalisierung:**
```python
# Längere Texte → höhere Scores (Problem)
# Lösung: Bereits im Modell eingebaut bei normalize_embeddings=True
```

**3. Re-Ranking mit Cross-Encoder:**
```python
from sentence_transformers import CrossEncoder

# Bi-Encoder (schnell): Top-100
bi_encoder = SentenceTransformer('model')
candidates = bi_encoder_search(query, k=100)

# Cross-Encoder (langsam aber genau): Top-10
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs)
top_10 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:10]
```

---

---

## Appendix: Model Catalog

### German Models

| Model | Architecture | Size | Dimensions | Training Data | Use Case |
|-------|-------------|------|------------|---------------|----------|
| `deepset/gbert-large` | BERT | 1.3GB | 1024 | German Wikipedia, News | General German |
| `deepset/gbert-base` | BERT | 400MB | 768 | German Wikipedia, News | Lightweight German |
| `GerMedBERT/medbert-512` | BERT | 400MB | 512 | Medical texts | German Medical |
| `smanjil/German-MedBERT` | BERT | 400MB | 768 | German medical corpora | German Medical (alt) |

### Multilingual Models

| Model | Architecture | Size | Dimensions | Languages | Use Case |
|-------|-------------|------|------------|-----------|----------|
| `intfloat/multilingual-e5-large` | E5 | 2.2GB | 1024 | 100+ | SOTA Multilingual |
| `intfloat/multilingual-e5-base` | E5 | 1.1GB | 768 | 100+ | Balanced Multilingual |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | MPNet | 280MB | 768 | 50+ | Lightweight Multilingual |

### English Models

| Model | Architecture | Size | Dimensions | Use Case |
|-------|-------------|------|------------|----------|
| `sentence-transformers/all-mpnet-base-v2` | MPNet | 420MB | 768 | English SOTA |
| `sentence-transformers/all-MiniLM-L6-v2` | MiniLM | 90MB | 384 | Fast & Lightweight |
| `dmis-lab/biobert-v1.1` | BERT | 400MB | 768 | Biomedical |
| `emilyalsentzer/Bio_ClinicalBERT` | BERT | 400MB | 768 | Clinical |

---

**Weitere Ressourcen:**
- [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=sentence-similarity)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence-Transformers Docs](https://www.sbert.net/)
