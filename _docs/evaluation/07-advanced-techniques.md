# Advanced Techniques

**Warum wichtig?** Basis-Metriken zeigen was funktioniert - Advanced Techniques zeigen wie du es systematisch verbessern kannst.

---

## RAGAS Framework

**Definition:** Automated RAG Assessment & Grading System - Framework für automatisierte RAG-Evaluation

**Installation:**
```bash
pip install ragas
```

**Code:**
```python
# ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # Keine Halluzinationen
    answer_relevancy,       # Beantwortet Frage
    context_precision,      # Relevante Chunks in Top-K
    context_recall,         # Alle relevanten Chunks gefunden
    context_relevancy,      # Chunks sind relevant zur Query
    answer_similarity,      # Ähnlich zu Ground Truth
)

# Dataset Format
from datasets import Dataset

data = {
    'question': ["Wie hoch ist der Energieverbrauch?"],
    'answer': ["Der Energieverbrauch beträgt 172 kWh pro Jahr."],
    'contexts': [["Energieverbrauch: 172 kWh pro Jahr",
                  "Gerät ist energieeffizient"]],
    'ground_truths': [["172 kWh pro Jahr"]]  # Optional
}

dataset = Dataset.from_dict(data)

# Evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
# Output:
# {
#   'faithfulness': 0.95,
#   'answer_relevancy': 0.92,
#   'context_precision': 0.88,
#   'context_recall': 1.0
# }
```

**Interpretation:**
- **faithfulness > 0.9**: Kaum Halluzinationen
- **answer_relevancy > 0.85**: Antworten sind relevant
- **context_precision > 0.8**: Retrieval ist präzise
- **context_recall > 0.9**: Retrieval findet alle relevanten Chunks

**Wann nutzen?**
- ✅ Schnelle End-to-End Evaluation
- ✅ Vergleich verschiedener RAG-Konfigurationen
- ✅ Continuous Evaluation in CI/CD

**Vorteile:**
- ✅ Automatisiert (kein manuelles Labeling für einige Metriken)
- ✅ Nutzt LLM-as-Judge intern
- ✅ Standard-Framework in der Community

**Nachteile:**
- ⚠️ Benötigt OpenAI/Anthropic API Key
- ⚠️ Kostet API Credits
- ⚠️ LLM-Judge kann selbst Fehler machen

**Target:**
- faithfulness: > 0.90
- answer_relevancy: > 0.85
- context_precision: > 0.80
- context_recall: > 0.90

---

## LLM-as-Judge

**Definition:** Nutze ein starkes LLM (GPT-4, Claude) um Antworten zu bewerten

**Code:**
```python
# anthropic
import anthropic
import json

def llm_judge_faithfulness(query, context, answer):
    """
    Nutze Claude als Judge für Faithfulness
    """
    client = anthropic.Anthropic(api_key="...")

    prompt = f"""
You are evaluating a RAG system's answer for faithfulness.

Query: {query}

Context:
{context}

Generated Answer:
{answer}

Task: Rate if the answer is fully grounded in the context.
- Score 1.0: All facts in answer are from context
- Score 0.5: Some facts missing in context
- Score 0.0: Answer contains hallucinations

Output format:
{{"score": <float>, "explanation": "<reasoning>"}}
"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse JSON response
    result = json.loads(response.content[0].text)
    return result['score'], result['explanation']

# Batch Evaluation
# python
def batch_llm_judge(results, metric='faithfulness'):
    scores = []
    for item in results:
        score, explanation = llm_judge_faithfulness(
            item['query'],
            item['context'],
            item['answer']
        )
        scores.append(score)

    return {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores)
    }
```

**Interpretation:**
- Mean Score > 0.9: System ist sehr faithful
- Mean Score 0.7-0.9: Gut, aber Verbesserungspotential
- Mean Score < 0.7: Problematisch - zu viele Halluzinationen

**Wann nutzen?**
- ✅ Wenn keine Ground Truth vorhanden
- ✅ Für komplexe Qualitätsbewertungen
- ✅ A/B Testing verschiedener Prompts/Models

**Vorteile:**
- ✅ Flexible (jede Metrik möglich)
- ✅ Erklärt Reasoning
- ✅ Keine Ground Truth nötig

**Nachteile:**
- ❌ Kostet API Credits
- ❌ Langsamer als regelbasierte Metriken
- ❌ LLM kann selbst Fehler machen

---

## Hard Negatives Mining

**Definition:** Finde Chunks die AUSSEHEN wie Antwort, aber FALSCH sind

**Code:**
```python
# numpy + sentence-transformers
def find_hard_negatives(query, positive_chunks, all_embeddings, k=5):
    """
    Finde Chunks die:
    1. Hohe Similarity zur Query haben
    2. ABER nicht in den relevanten Chunks sind
    """
    query_emb = model.encode(query)
    scores = all_embeddings @ query_emb

    # Top-K ähnlichste
    top_indices = np.argsort(scores)[::-1][:k*3]

    # Filtere positive Chunks raus
    positive_ids = set(positive_chunks)
    hard_negatives = [i for i in top_indices
                      if i not in positive_ids][:k]

    return hard_negatives

# Use Case: Testen ob System falsche Chunks unterscheiden kann
query = "Energieverbrauch HMFvh 4001"
positive = [42, 43]  # Chunks über HMFvh 4001

hard_negs = find_hard_negatives(query, positive, embeddings)
# → Chunks über HMFvh 5501 (anderes Modell, aber ähnliche Specs!)

# Prüfe: Ranked System positive höher als hard negatives?
```

**Interpretation:**
- System rankt Positives > Hard Negatives: Gut - erkennt Unterschiede
- System rankt Hard Negatives höher: Problematisch - verwechselt ähnliche Inhalte

**Wann nutzen?**
- ✅ Testen ob System ähnliche Produkte unterscheiden kann
- ✅ Model Training (falls du Fine-Tuning machst)
- ✅ Schwachstellen identifizieren

**Target:** Positive Chunks sollten im Schnitt 0.1-0.2 höhere Similarity haben als Hard Negatives

---

## A/B Testing

**Definition:** Vergleiche zwei System-Versionen mit echten Usern

**Setup:**
```python
# python
import random

class ABTestRAG:
    def __init__(self, system_a, system_b):
        self.system_a = system_a
        self.system_b = system_b
        self.results = {'A': [], 'B': []}

    def query(self, query, user_id):
        # 50/50 Split
        variant = 'A' if hash(user_id) % 2 == 0 else 'B'
        system = self.system_a if variant == 'A' else self.system_b

        answer = system.query(query)

        self.results[variant].append({
            'query': query,
            'answer': answer,
            'user_id': user_id
        })

        return answer, variant

    def analyze(self):
        """Vergleiche Metriken zwischen A und B"""
        metrics_a = calculate_metrics(self.results['A'])
        metrics_b = calculate_metrics(self.results['B'])

        return {
            'A': metrics_a,
            'B': metrics_b,
            'winner': 'A' if metrics_a['score'] > metrics_b['score'] else 'B'
        }

# Example
system_a = RAGSystem(model='gbert-large', top_k=3)
system_b = RAGSystem(model='gbert-large', top_k=5)  # Unterschied: Top-K

ab_test = ABTestRAG(system_a, system_b)

# Production: User queries
for query, user_id in user_queries:
    answer, variant = ab_test.query(query, user_id)
    # Track user feedback (thumbs up/down)

# Nach 1000 Queries:
results = ab_test.analyze()
print(f"Winner: {results['winner']}")
```

**Was testen?**
- Unterschiedliche Models (GBERT vs E5)
- Unterschiedliche Top-K (3 vs 5)
- Unterschiedliche Prompts
- Mit/Ohne Reranking

**Statistische Signifikanz:**
```python
# scipy
from scipy import stats

def is_significant(results_a, results_b, alpha=0.05):
    """T-Test für statistische Signifikanz"""
    t_stat, p_value = stats.ttest_ind(results_a, results_b)
    return p_value < alpha, p_value

# Beispiel
scores_a = [0.85, 0.90, 0.88, 0.92, 0.87]  # System A Scores
scores_b = [0.78, 0.82, 0.80, 0.85, 0.79]  # System B Scores

significant, p_value = is_significant(scores_a, scores_b)
print(f"Significant: {significant}, p-value: {p_value:.4f}")
```

**Wann nutzen?**
- ✅ **Production** - echter User-Feedback
- ✅ Wichtige System-Änderungen validieren
- ✅ Wenn Metriken alleine nicht ausreichen

**Target:** Mindestens 500 Queries pro Variante für reliable Ergebnisse

---

## Failure Analysis

**Definition:** Systematisch Fehler analysieren um Patterns zu finden

**Code:**
```python
# python
import re

def failure_analysis(results, ground_truth, threshold=0.5):
    """
    Finde Queries wo System schlecht performed
    """
    failures = []

    for query_id, retrieved in results.items():
        relevant = ground_truth[query_id]
        precision = precision_at_k(retrieved, relevant, k=5)

        if precision < threshold:
            failures.append({
                'query_id': query_id,
                'query_text': get_query_text(query_id),
                'precision': precision,
                'retrieved': retrieved[:5],
                'relevant': relevant
            })

    # Kategorisiere Failures
    categories = {
        'short_query': [],      # Query < 5 Wörter
        'technical_terms': [],  # Enthält Produktcodes
        'multi_intent': [],     # Multiple Aspekte gefragt
        'ambiguous': [],        # Unklar formuliert
    }

    for failure in failures:
        query = failure['query_text']
        if len(query.split()) < 5:
            categories['short_query'].append(failure)
        if re.search(r'[A-Z]{2,}\d+', query):  # z.B. "HMFvh4001"
            categories['technical_terms'].append(failure)
        # ... weitere Kategorisierung

    return categories

# Nutzen:
failures = failure_analysis(results, ground_truth)
print(f"Short queries: {len(failures['short_query'])} failures")
print(f"Technical terms: {len(failures['technical_terms'])} failures")

# → Erkenne: "Queries mit Produktcodes performen schlecht"
# → Action: Query Preprocessing verbessern
```

**Interpretation:**
- Viele Failures in einer Kategorie: Systematisches Problem
- Gleichmäßig verteilt: Model generell schwach
- Einzelne Outliers: Edge Cases

**Wann nutzen?**
- ✅ Nach jeder Evaluation (verstehe WO System versagt)
- ✅ Feature Priorisierung (was fixen zuerst?)
- ✅ Model Improvement Planning

**Action Items basierend auf Failure Patterns:**
- **Short Queries**: Query Expansion implementieren
- **Technical Terms**: Spezial-Preprocessing für Codes
- **Multi-Intent**: Multi-Step Retrieval
- **Ambiguous**: Clarification Dialog

---

## Cross-Encoder Reranking

**Definition:** Nutze Cross-Encoder um Initial Retrieval zu verbessern

**Code:**
```python
# sentence-transformers
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np

# Initial Retrieval (Bi-Encoder, schnell)
bi_encoder = SentenceTransformer('deepset/gbert-large')
query_emb = bi_encoder.encode(query)
scores = embeddings @ query_emb
top_20 = np.argsort(scores)[::-1][:20]  # Top-20 candidates

# Reranking (Cross-Encoder, langsamer aber genauer)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score Query-Chunk Paare
pairs = [(query, chunks[i]) for i in top_20]
rerank_scores = cross_encoder.predict(pairs)

# Top-5 nach Reranking
top_5_indices = np.argsort(rerank_scores)[::-1][:5]
final_chunks = [top_20[i] for i in top_5_indices]

# Evaluate: Precision before vs after reranking
precision_before = precision_at_k(top_20[:5], relevant, 5)
precision_after = precision_at_k(final_chunks, relevant, 5)

print(f"Precision before: {precision_before:.2f}")
print(f"Precision after: {precision_after:.2f}")
print(f"Improvement: {precision_after - precision_before:.2f}")
```

**Formel:**
```
Bi-Encoder Score: cos(query_emb, chunk_emb)
Cross-Encoder Score: model([query, chunk])  # Joint encoding

Bi-Encoder: Fast (independent encoding)
Cross-Encoder: Slow (pairwise comparison) but more accurate
```

**Interpretation:**
- Improvement > 0.1: Reranking lohnt sich definitiv
- Improvement 0.05-0.1: Moderate Verbesserung
- Improvement < 0.05: Reranking bringt wenig

**Wann nutzen?**
- ✅ Wenn Latenz okay ist (+500ms)
- ✅ Precision ist wichtiger als Speed
- ✅ Top-20 Candidates sind gut genug (hoher Recall)

**Target:** Precision Improvement > 0.10

---

## Query Expansion Testing

**Definition:** Erweitere Query mit Synonymen/verwandten Begriffen

**Code:**
```python
# python
def expand_query(query):
    """
    Erweitere Query mit Synonymen
    """
    expansions = {
        'Kühlschrank': ['Kühlgerät', 'Refrigerator', 'Kühlschrank'],
        'Energieverbrauch': ['kWh', 'Stromverbrauch', 'Jahresverbrauch'],
        'Alarm': ['Warnung', 'Alarmierung', 'Benachrichtigung'],
    }

    tokens = query.split()
    expanded = []

    for token in tokens:
        if token in expansions:
            expanded.extend(expansions[token])
        else:
            expanded.append(token)

    return ' '.join(expanded)

# A/B Test
query_original = "Energieverbrauch Kühlschrank"
query_expanded = expand_query(query_original)
# → "kWh Stromverbrauch Jahresverbrauch Kühlgerät Refrigerator Kühlschrank"

results_original = retrieve(query_original)
results_expanded = retrieve(query_expanded)

# Compare Precision
precision_original = precision_at_k(results_original, relevant, 5)
precision_expanded = precision_at_k(results_expanded, relevant, 5)

print(f"Original: {precision_original:.2f}")
print(f"Expanded: {precision_expanded:.2f}")
```

**Interpretation:**
- Precision verbessert: Query Expansion hilft
- Precision verschlechtert: Query Drift (zu viele irrelevante Begriffe)

**Wann nutzen?**
- ✅ Bei kurzen Queries (< 5 Wörter)
- ✅ Bei domain-spezifischen Synonymen
- ✅ Wenn Retrieval zu spezifisch ist (hohe Precision, niedrige Recall)

**Risiko:** Query Drift (zu viele Expansions → irrelevante Results)

**Target:** Precision Improvement ohne Recall-Verlust

---

## Hybrid Search Evaluation

**Definition:** Kombiniere Semantic Search + Keyword Search (BM25)

**Code:**
```python
# rank-bm25
from rank_bm25 import BM25Okapi
import numpy as np

# Prepare BM25
tokenized_docs = [doc.split() for doc in all_chunks]
bm25 = BM25Okapi(tokenized_docs)

def hybrid_search(query, alpha=0.5):
    """
    Kombiniere Semantic (alpha) + BM25 (1-alpha)
    """
    # Semantic Scores
    query_emb = model.encode(query)
    semantic_scores = embeddings @ query_emb
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())

    # BM25 Scores
    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

    # Combine
    hybrid_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

    top_k = np.argsort(hybrid_scores)[::-1][:10]
    return top_k

# Evaluate different alphas
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results = hybrid_search(query, alpha=alpha)
    precision = precision_at_k(results, relevant, 5)
    print(f"Alpha {alpha}: Precision = {precision:.2f}")

# Find optimal alpha
```

**Formel:**
```
Hybrid_Score = α × Semantic_Score + (1-α) × BM25_Score

α = 0: Pure Keyword Search
α = 0.5: Equal Weight
α = 1: Pure Semantic Search
```

**Interpretation:**
- Optimal α < 0.5: Keywords wichtiger (technische Queries)
- Optimal α ≈ 0.5: Beide gleich wichtig
- Optimal α > 0.5: Semantik wichtiger (natürliche Fragen)

**Wann nutzen?**
- ✅ Queries mit spezifischen Keywords ("HMFvh 4001")
- ✅ Exakte Matches wichtig (Produktcodes, Normen)
- ✅ Semantic alleine nicht gut genug

**Target:** Precision Improvement > 0.05 vs. pure Semantic

---

## Advanced Techniques: Summary

| Technique | Use Case | Complexity | Impact |
|-----------|----------|------------|--------|
| **RAGAS** | End-to-End Evaluation | Low | High |
| **LLM-as-Judge** | Quality ohne Ground Truth | Medium | High |
| **Hard Negatives** | Model Robustness Testing | Medium | Medium |
| **A/B Testing** | Production Validation | Low | High |
| **Failure Analysis** | Problem Identification | Low | High |
| **Cross-Encoder** | Precision Improvement | Medium | Medium-High |
| **Query Expansion** | Recall Improvement | Low | Medium |
| **Hybrid Search** | Balanced Retrieval | Medium | Medium-High |

**Empfohlene Reihenfolge:**
1. **RAGAS** - Quick Baseline Evaluation
2. **Failure Analysis** - Verstehe wo System versagt
3. **A/B Testing** - Teste Improvements in Production
4. **Cross-Encoder** - Wenn Precision zu niedrig
5. **Hybrid Search** - Wenn Keyword-Matches wichtig
