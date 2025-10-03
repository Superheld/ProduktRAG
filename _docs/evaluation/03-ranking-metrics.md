# Ranking Metrics

**Warum wichtig?** Ranking bestimmt welche Chunks ans LLM gehen - schlechtes Ranking = schlechte Antworten.

---

## Precision@K

**Definition:** Von den Top-K Ergebnissen, wie viele sind relevant?

**Formel:**
```
Precision@K = (Anzahl relevante Docs in Top-K) / K
Range: [0, 1], höher = besser
```

**Code:**
```python
# python
def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant))
    return relevant_in_top_k / k

# Beispiel
retrieved = [1, 5, 3, 8, 2, 9, 7, 4, 6, 10]
relevant = {1, 2, 3}

print(precision_at_k(retrieved, relevant, k=5))  # 3/5 = 0.6
print(precision_at_k(retrieved, relevant, k=3))  # 2/3 = 0.67
```

**Interpretation:**
- `Precision@3 = 1.0`: Alle Top-3 sind relevant (perfekt!)
- `Precision@5 = 0.6`: 3 von 5 sind relevant (okay)
- `Precision@10 = 0.3`: Nur 3 von 10 relevant (schlecht)

**Wann nutzen?**
- ✅ Wenn nur Top-K ans LLM gesendet wird
- ✅ User sieht nur erste Ergebnisse
- ✅ Token-Budget limitiert (nur Top-3 nutzbar)
- ❌ Ignoriert Reihenfolge innerhalb Top-K

**Target:**
- Precision@3: > 0.80
- Precision@5: > 0.70
- Precision@10: > 0.60

---

## Recall@K

**Definition:** Von allen relevanten Docs, wie viele wurden in Top-K gefunden?

**Formel:**
```
Recall@K = (Anzahl gefundene relevante Docs) / (Total relevante Docs)
Range: [0, 1], höher = besser
```

**Code:**
```python
# python
def recall_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant))
    return relevant_in_top_k / len(relevant)

# Beispiel
retrieved = [1, 5, 3, 8, 2]
relevant = {1, 2, 3}
print(recall_at_k(retrieved, relevant, k=5))  # 3/3 = 1.0 (alle gefunden!)
```

**Interpretation:**
- `Recall@10 = 1.0`: Alle relevanten Docs in Top-10 (perfekt!)
- `Recall@5 = 0.6`: Nur 60% der relevanten Docs gefunden
- `Recall@3 = 0.33`: Nur 1 von 3 relevanten Docs in Top-3

**Wann nutzen?**
- ✅ Wichtig bei wenigen relevanten Docs (keine verpassen!)
- ✅ Multi-hop Reasoning (mehrere Chunks nötig)
- ✅ Retrieval-System Grundsatz-Evaluation
- ❌ Nicht relevant wenn nur Top-1 zählt

**Target:**
- Recall@5: > 0.70
- Recall@10: > 0.90
- Recall@20: > 0.95

---

## F1-Score@K

**Definition:** Harmonisches Mittel aus Precision und Recall

**Formel:**
```
F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
Range: [0, 1], höher = besser
```

**Code:**
```python
# python
def f1_at_k(retrieved, relevant, k):
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
```

**Interpretation:**
- Balanciert Precision und Recall
- Niedriger wenn eine Metrik sehr schlecht ist
- Höher wenn beide Metriken gut sind

**Wann nutzen?**
- ✅ Trade-off zwischen Precision und Recall wichtig
- ✅ Vergleich verschiedener Systeme (eine Metrik statt zwei)
- ❌ Wenn Precision >> Recall wichtig (z.B. nur Top-3 zählt)

**Target:** F1@5 > 0.70

---

## Mean Reciprocal Rank (MRR)

**Definition:** Durchschnittliche Position des ERSTEN relevanten Ergebnisses

**Formel:**
```
RR = 1 / rank(first_relevant_doc)
MRR = average(RR) über alle Queries
Range: [0, 1], höher = besser
```

**Code:**
```python
# python
def reciprocal_rank(retrieved, relevant):
    """
    Args:
        retrieved: List[int] - Retrieved doc IDs (ranked)
        relevant: Set[int] - Relevant doc IDs
    Returns:
        float - Reciprocal rank (0 if none found)
    """
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(results, ground_truth):
    """
    Args:
        results: Dict[query_id, List[doc_ids]]
        ground_truth: Dict[query_id, Set[relevant_doc_ids]]
    Returns:
        float - MRR
    """
    rr_scores = []
    for query_id, retrieved in results.items():
        relevant = ground_truth[query_id]
        rr_scores.append(reciprocal_rank(retrieved, relevant))
    return sum(rr_scores) / len(rr_scores)

# Beispiel
retrieved = [8, 5, 1, 3, 2]
relevant = {1, 2, 3}
print(reciprocal_rank(retrieved, relevant))  # 1/3 = 0.333 (3. Position)

retrieved = [1, 5, 3, 8, 2]
print(reciprocal_rank(retrieved, relevant))  # 1/1 = 1.0 (1. Position!)
```

**Interpretation:**
- `MRR = 1.0`: Erstes Ergebnis ist immer relevant (perfekt!)
- `MRR = 0.5`: Durchschnittlich auf Position 2
- `MRR = 0.33`: Durchschnittlich auf Position 3
- `MRR = 0.1`: Durchschnittlich auf Position 10 (schlecht)

**Wann nutzen?**
- ✅ **Wichtigste Metrik** wenn nur Top-1 zählt
- ✅ User klickt meist nur erstes Ergebnis
- ✅ LLM nutzt hauptsächlich ersten Chunk
- ✅ Single-Fact Retrieval (eine Antwort gesucht)

**Target:**
- MRR > 0.8 (erstes relevantes Doc meist in Top-2)
- MRR > 0.9 (meist auf Position 1)

---

## Mean Average Precision (MAP)

**Definition:** Durchschnitt der Precision-Werte an jeder Position eines relevanten Docs

**Formel:**
```
AP = (Σ Precision@k * rel(k)) / num_relevant
MAP = average(AP) über alle Queries
```

**Code:**
```python
# python
def average_precision(retrieved, relevant):
    """
    Args:
        retrieved: List[int] - Retrieved docs (ranked)
        relevant: Set[int] - Relevant docs
    Returns:
        float - Average Precision
    """
    if len(relevant) == 0:
        return 0.0

    precision_sum = 0.0
    num_relevant_found = 0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / i
            precision_sum += precision_at_i

    return precision_sum / len(relevant)

def mean_average_precision(results, ground_truth):
    ap_scores = []
    for query_id, retrieved in results.items():
        relevant = ground_truth[query_id]
        ap_scores.append(average_precision(retrieved, relevant))
    return sum(ap_scores) / len(ap_scores)

# Beispiel
retrieved = [1, 5, 3, 8, 2, 9]
relevant = {1, 3, 2}
# Position 1: relevant (Prec = 1/1 = 1.0)
# Position 3: relevant (Prec = 2/3 = 0.67)
# Position 5: relevant (Prec = 3/5 = 0.6)
# AP = (1.0 + 0.67 + 0.6) / 3 = 0.756
print(average_precision(retrieved, relevant))  # 0.756
```

**Wann nutzen?**
- ✅ Berücksichtigt Reihenfolge UND Anzahl
- ✅ Standard in Information Retrieval Research
- ✅ Vergleich verschiedener Retrieval-Systeme

---

## Normalized Discounted Cumulative Gain (NDCG)

**Definition:** Berücksichtigt Ranking-Qualität UND abgestufte Relevanz (0, 1, 2, 3)

**Formel:**
```
DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@K = DCG@K / IDCG@K (ideal DCG)
Range: [0, 1], höher = besser
```

**Code:**
```python
# numpy
import numpy as np

def dcg_at_k(relevances, k):
    """
    Args:
        relevances: List[int] - Relevance scores (0-3) for each position
        k: int - Cutoff
    Returns:
        float - DCG score
    """
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0

    # Positions start at 1
    positions = np.arange(1, len(relevances) + 1)

    # DCG formula: (2^rel - 1) / log2(position + 1)
    return np.sum((2**relevances - 1) / np.log2(positions + 1))

def ndcg_at_k(retrieved_relevances, ideal_relevances, k):
    """
    Args:
        retrieved_relevances: List[int] - Relevance scores in retrieved order
        ideal_relevances: List[int] - Relevance scores in ideal order (sorted desc)
        k: int - Cutoff
    Returns:
        float - NDCG score
    """
    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

# Beispiel
# Query: "Energieverbrauch HMFvh 4001?"
# Retrieved docs mit Relevanz-Scores:
retrieved = [3, 0, 2, 1, 0]  # 3=perfect, 2=good, 1=ok, 0=irrelevant
ideal = [3, 2, 1, 0, 0]      # Optimal sortiert

print(ndcg_at_k(retrieved, ideal, k=5))  # ~0.95 (gut!)

# Schlechtes Ranking:
retrieved_bad = [0, 1, 0, 2, 3]  # Beste Docs ganz hinten
print(ndcg_at_k(retrieved_bad, ideal, k=5))  # ~0.65 (schlecht)
```

**Relevanz-Skala:**
- `3`: Perfekte Antwort (beantwortet Query vollständig)
- `2`: Sehr relevant (hilfreiche Info, aber unvollständig)
- `1`: Teilweise relevant (tangential relevant)
- `0`: Irrelevant

**Interpretation:**
- `NDCG@5 = 1.0`: Perfektes Ranking
- `NDCG@5 = 0.9`: Sehr gutes Ranking (kleine Fehler)
- `NDCG@5 = 0.7`: Okay (einige relevante Docs zu weit unten)
- `NDCG@5 = 0.5`: Schlecht (Ranking zufällig)

**Wann nutzen?**
- ✅ **Gold Standard** für Ranking-Evaluation
- ✅ Wenn Relevanz abgestuft ist (nicht nur binary)
- ✅ Position ist wichtig (frühere Docs zählen mehr)
- ✅ Vergleich verschiedener Ranking-Algorithmen

**Target:**
- NDCG@5 > 0.85
- NDCG@10 > 0.80

---

## Hit Rate@K

**Definition:** Prozent der Queries mit mindestens 1 relevantem Doc in Top-K

**Code:**
```python
# python
def hit_rate_at_k(retrieved, relevant, k):
    """Binary: 1 if any relevant doc in top-k, else 0"""
    top_k = set(retrieved[:k])
    return 1.0 if len(top_k & relevant) > 0 else 0.0

def mean_hit_rate(results, ground_truth, k):
    hits = []
    for query_id, retrieved in results.items():
        relevant = ground_truth[query_id]
        hits.append(hit_rate_at_k(retrieved, relevant, k))
    return sum(hits) / len(hits)
```

**Wann nutzen?**
- ✅ Minimale Anforderung: "Ist überhaupt was relevantes dabei?"
- ✅ Schnelle Sanity Check

**Target:** Hit Rate@10 > 0.95

---

## Welche Metrik wann?

| Use Case | Empfohlene Metrik | Warum? |
|----------|------------------|--------|
| **LLM nutzt Top-1** | MRR | Nur erstes Ergebnis zählt |
| **LLM nutzt Top-3** | Precision@3, NDCG@3 | Qualität der Top-3 wichtig |
| **Multiple relevant docs nötig** | Recall@10, MAP | Keine verpassen |
| **Abgestufte Relevanz** | NDCG@K | Beste comprehensive Metrik |
| **Quick Sanity Check** | Hit Rate@10 | Ist überhaupt was relevantes dabei? |
| **A/B Testing** | MRR + NDCG@5 | Kombiniert Ranking + Quality |
