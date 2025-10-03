# RAG Evaluation Guide

Thematisch organisiertes Nachschlagewerk für RAG Evaluation Metrics & Techniken.

---

## 📑 Inhaltsverzeichnis

### [1. Chunk Quality Evaluation](./01-chunk-quality.md)
Datenqualität prüfen BEVOR du embeddest
- Length Statistics
- Outlier Detection
- Empty/Null Checks
- Metadata Completeness
- Content Quality Patterns
- Quality Scorecard

### [2. Similarity Measures](./02-similarity-measures.md)
Wie vergleiche ich Vektoren?
- Cosine Similarity ⭐
- Dot Product
- Euclidean Distance (L2)
- Manhattan Distance (L1)

### [3. Ranking Metrics](./03-ranking-metrics.md)
Wie gut ist mein Retrieval?
- Precision@K ⭐
- Recall@K
- F1-Score@K
- Mean Reciprocal Rank (MRR) ⭐
- Mean Average Precision (MAP)
- NDCG ⭐
- Hit Rate@K

### [4. Quality Metrics](./04-quality-metrics.md)
Wie gut sind meine LLM-Antworten?
- Faithfulness / Groundedness ⭐
- Answer Relevance
- Correctness / Factuality
- Completeness
- Conciseness
- Citation Accuracy

### [5. Embedding Evaluation](./05-embedding-evaluation.md)
Wie gut sind meine Embeddings?
- Semantic Similarity Tests
- Clustering Quality
- Outlier Detection
- Cross-Lingual Test
- Dimensionality Analysis

### [6. System Metrics](./06-system-metrics.md)
Production-Performance
- Latency (Response Time) ⭐
- Throughput
- Cost per Query
- Cache Hit Rate
- Error Rate
- User Satisfaction Metrics

### [7. Advanced Techniques](./07-advanced-techniques.md)
Spezielle Evaluationsmethoden
- RAGAS Framework ⭐
- LLM-as-Judge
- Hard Negatives Mining
- A/B Testing
- Failure Analysis
- Cross-Encoder Reranking
- Query Expansion Testing
- Hybrid Search Evaluation

---

## 🎯 Quick Reference: Welche Metrik wann?

| Situation | Empfohlene Metrik(en) | Guide |
|-----------|----------------------|--------|
| **Chunking validieren** | Length Stats, Outliers, Quality Score | [Chunk Quality](./01-chunk-quality.md) |
| **Retrieval testen (ohne LLM)** | Precision@5, Recall@10, MRR | [Ranking Metrics](./03-ranking-metrics.md) |
| **Ranking optimieren** | NDCG@K, MAP | [Ranking Metrics](./03-ranking-metrics.md) |
| **LLM Antworten evaluieren** | Faithfulness, Answer Relevance | [Quality Metrics](./04-quality-metrics.md) |
| **Production Monitoring** | Latency, Throughput, Error Rate | [System Metrics](./06-system-metrics.md) |
| **A/B Test zweier Systeme** | MRR + NDCG@5 + User Thumbs-Up | [Advanced](./07-advanced-techniques.md) |
| **Embedding-Qualität prüfen** | Semantic Similarity Tests, Clustering | [Embedding](./05-embedding-evaluation.md) |
| **Cost Optimization** | Cost per Query, Cache Hit Rate | [System Metrics](./06-system-metrics.md) |

---

## 🔄 Evaluations-Workflow

### Phase 0: Chunk Quality (VOR Embedding)
```python
# 1. Load Chunks
df = pd.read_json('chunks.jsonl', lines=True)

# 2. Quality Checks
from evaluation import chunk_quality_score
score = chunk_quality_score(df)  # Target: > 95/100

# 3. Fix Issues
# - Entferne leere Chunks
# - Merge zu kurze Chunks
# - Split zu lange Chunks
```
→ **[Siehe Chunk Quality Guide](./01-chunk-quality.md)**

### Phase 1: Retrieval Evaluation (NACH Embedding)
```python
# 1. Ground Truth erstellen
annotations = annotate_queries(test_queries, chunks)

# 2. Retrieval durchführen
results = run_retrieval(test_queries, embeddings)

# 3. Metriken berechnen
precision = precision_at_k(results, annotations, k=5)
mrr = mean_reciprocal_rank(results, annotations)
ndcg = ndcg_at_k(results, annotations, k=5)

# 4. Failure Analysis
failures = find_failed_queries(results, annotations)
```
→ **[Siehe Ranking Metrics Guide](./03-ranking-metrics.md)**

### Phase 2: Generation Evaluation (mit LLM)
```python
# 1. Generate Answers
answers = []
for query in test_queries:
    chunks = retrieve(query, k=3)
    answer = llm_generate(query, chunks)
    answers.append(answer)

# 2. Automated Evaluation (RAGAS)
from ragas import evaluate
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

# 3. LLM-as-Judge
faithfulness_scores = llm_judge_batch(answers, contexts)
```
→ **[Siehe Quality Metrics Guide](./04-quality-metrics.md)**
→ **[Siehe Advanced Techniques Guide](./07-advanced-techniques.md)**

### Phase 3: Production Monitoring
```python
# 1. Logging
log_query(query, answer, latency, user_id, timestamp)

# 2. Metrics Collection
collect_metrics(['latency', 'error_rate', 'cache_hit_rate'])

# 3. User Feedback
track_feedback(['thumbs_up', 'thumbs_down'])

# 4. Alerting
if error_rate > 0.05:
    alert("High error rate!")
```
→ **[Siehe System Metrics Guide](./06-system-metrics.md)**

---

## 📚 Tool-Übersicht

Die Guides verwenden folgende Tools:

| Tool | Use Cases | Installation |
|------|-----------|--------------|
| **pandas** | Chunk Analysis, Data Processing | `pip install pandas` |
| **numpy** | Vector Operations, Statistics | `pip install numpy` |
| **sklearn** | Similarity, Clustering, Metrics | `pip install scikit-learn` |
| **sentence-transformers** | Embeddings, Similarity | `pip install sentence-transformers` |
| **ragas** | Automated RAG Evaluation | `pip install ragas` |
| **anthropic / openai** | LLM-as-Judge | `pip install anthropic openai` |
| **rank-bm25** | Hybrid Search (BM25) | `pip install rank-bm25` |

---

## 🎓 Lern-Pfad

**Neu bei RAG Evaluation?** Empfohlene Reihenfolge:

1. **[Chunk Quality](./01-chunk-quality.md)** - Basics: Datenqualität verstehen
2. **[Similarity Measures](./02-similarity-measures.md)** - Wie funktioniert Ähnlichkeit?
3. **[Ranking Metrics](./03-ranking-metrics.md)** - Retrieval evaluieren (wichtigste Metriken)
4. **[Quality Metrics](./04-quality-metrics.md)** - LLM-Antworten bewerten
5. **[System Metrics](./06-system-metrics.md)** - Production-ready machen
6. **[Embedding Evaluation](./05-embedding-evaluation.md)** - Deep dive: Embedding-Qualität
7. **[Advanced Techniques](./07-advanced-techniques.md)** - Spezielle Methoden & Frameworks

---

*Last updated: 2025-10-03*
